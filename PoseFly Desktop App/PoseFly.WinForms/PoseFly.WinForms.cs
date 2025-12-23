using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
using System.Diagnostics;
using System.Net.NetworkInformation;

namespace PoseFly.WinForms
{
    public partial class Form1 : Form
    {
        private TcpClient? _client;
        private NetworkStream? _stream;
        private CancellationTokenSource? _cts;
        private Task? _readerTask;

        // One terminal only
        private Process? _serverTerminal;

        private const string Host = "127.0.0.1";
        private const int Port = 8765;

        public Form1()
        {
            InitializeComponent();

            UpdateStatus("Idle");

            btnBrowse.Click += BtnBrowse_Click;
            btnStart.Click += BtnStart_Click;
            btnStop.Click += BtnStop_Click;
            btnQuit.Click += BtnQuit_Click;
            this.FormClosing += Form1_FormClosing;

            // Optional: live updates while running (safe even if not connected yet)
            numIso.ValueChanged += async (_, __) => await SendRollingUpdateAsync();
            numShutterHz.ValueChanged += async (_, __) => await SendRollingUpdateAsync();
        }

        // --------- Event Handlers ---------

        private void BtnBrowse_Click(object? sender, EventArgs e) => BrowseOutput();
        private async void BtnStart_Click(object? sender, EventArgs e) => await StartAsync();
        private async void BtnStop_Click(object? sender, EventArgs e) => await StopAsync();

        private async void BtnQuit_Click(object? sender, EventArgs e)
        {
            await StopAsync();
            Close();
        }

        private async void Form1_FormClosing(object? sender, FormClosingEventArgs e) => await StopAsync();

        // --------- Start Python server in ONE terminal ---------

        private void StartServerInTerminal()
        {
            if (_serverTerminal != null && !_serverTerminal.HasExited)
                return;

            var exeDir = AppContext.BaseDirectory;
            var pyDir = Path.Combine(exeDir, "PyBackend");
            var script = Path.Combine(pyDir, "posefly_server.py");

            if (!Directory.Exists(pyDir))
                throw new DirectoryNotFoundException("Missing folder:\n" + pyDir);

            if (!File.Exists(script))
                throw new FileNotFoundException("Missing script:\n" + script);

            var psi = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = "/k python posefly_server.py",
                WorkingDirectory = pyDir,
                UseShellExecute = false,
                CreateNoWindow = false
            };

            _serverTerminal = new Process();
            _serverTerminal.StartInfo = psi;
            _serverTerminal.Start();
        }

        // IMPORTANT: do NOT probe readiness by CONNECTING (single-client server).
        // Instead, check if port is LISTENING.
        private bool IsPortListening(int port)
        {
            try
            {
                var props = IPGlobalProperties.GetIPGlobalProperties();
                var listeners = props.GetActiveTcpListeners();
                foreach (var ep in listeners)
                {
                    if (ep.Port == port) return true;
                }
                return false;
            }
            catch
            {
                return false;
            }
        }

        private async Task WaitForServerAsync(int timeoutMs)
        {
            var start = Environment.TickCount;

            while (Environment.TickCount - start < timeoutMs)
            {
                if (IsPortListening(Port))
                    return;

                await Task.Delay(150);
            }

            throw new Exception("Python server did not start (port 8765 not LISTENING).");
        }

        // --------- Existing UI Helpers ---------

        private void BrowseOutput()
        {
            using var dlg = new SaveFileDialog
            {
                Filter = "MP4 Video (*.mp4)|*.mp4|All Files (*.*)|*.*",
                DefaultExt = "mp4",
                FileName = Path.GetFileName(txtOutput.Text)
            };

            if (dlg.ShowDialog(this) == DialogResult.OK)
                txtOutput.Text = dlg.FileName;
        }

        private double GetFpsSafe()
        {
            if (double.TryParse(txtFps.Text.Trim(), out var fps) && fps > 0) return fps;
            return 10.0;
        }

        private object BuildPayload()
        {
            return new
            {
                camera_index = (int)numCamera.Value,
                use_dshow = true,
                fps = GetFpsSafe(),
                output_path = txtOutput.Text.Trim(),
                save_video = chkSave.Checked,

                // Rolling shutter controls
                iso = (int)numIso.Value,
                shutter_hz = (double)numShutterHz.Value,

                toggles = new
                {
                    drone = chkDrone.Checked,
                    angle = chkAngle.Checked,
                    distance = chkDistance.Checked,
                    led = chkLed.Checked
                }
            };
        }

        // --------- Networking ---------

        private async Task ConnectAsync()
        {
            if (_client != null) return;

            _client = new TcpClient();
            await _client.ConnectAsync(Host, Port);

            _stream = _client.GetStream();

            _cts = new CancellationTokenSource();
            _readerTask = Task.Run(() => ReaderLoop(_cts.Token));
        }

        private async Task SendJsonAsync(object msg)
        {
            if (_stream == null) return;

            var json = JsonSerializer.Serialize(msg);
            var bytes = Encoding.UTF8.GetBytes(json + "\n");
            await _stream.WriteAsync(bytes, 0, bytes.Length);
        }

        private async Task SendRollingUpdateAsync()
        {
            // only send if connected
            if (_client == null || _stream == null) return;

            try
            {
                await SendJsonAsync(new
                {
                    cmd = "UPDATE",
                    payload = new
                    {
                        iso = (int)numIso.Value,
                        shutter_hz = (double)numShutterHz.Value
                    }
                });
            }
            catch
            {
                // ignore update errors (stopping, disconnect, etc.)
            }
        }

        private async Task StartAsync()
        {
            try
            {
                StartServerInTerminal();
                await WaitForServerAsync(10000);

                await ConnectAsync();
                await SendJsonAsync(new { cmd = "START", payload = BuildPayload() });

                UpdateStatus("Starting...");
            }
            catch (Exception ex)
            {
                UpdateStatus("Error");
                MessageBox.Show(this, ex.Message, "Start Failed",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private async Task StopAsync()
        {
            try
            {
                // 1) Best-effort tell server to stop
                try
                {
                    if (_client != null)
                        await SendJsonAsync(new { cmd = "STOP", payload = new { } });
                }
                catch { }

                // 2) Stop reader loop
                try { _cts?.Cancel(); } catch { }

                // 3) Wait for reader to finish
                try { if (_readerTask != null) await _readerTask; } catch { }

                // 4) Close network resources
                try { _stream?.Close(); } catch { }
                try { _client?.Close(); } catch { }

                _stream = null;
                _client = null;

                try { _cts?.Dispose(); } catch { }
                _cts = null;
                _readerTask = null;

                // Clear live view (PictureBox)
                try
                {
                    BeginInvoke(new Action(() =>
                    {
                        var old = pic.Image;
                        pic.Image = null;
                        old?.Dispose();
                    }));
                }
                catch { }

                // 5) Close the terminal (and python child)
                try
                {
                    if (_serverTerminal != null && !_serverTerminal.HasExited)
                        _serverTerminal.Kill(entireProcessTree: true);
                }
                catch
                {
                    try { _serverTerminal?.Kill(); } catch { }
                }

                _serverTerminal = null;

                UpdateStatus("Stopped");
            }
            catch
            {
                // ignore shutdown errors
            }
        }

        private void ReaderLoop(CancellationToken ct)
        {
            if (_stream == null) return;

            using var reader = new StreamReader(_stream, Encoding.UTF8, false, 1 << 16, leaveOpen: true);

            while (!ct.IsCancellationRequested)
            {
                string? line;
                try
                {
                    line = reader.ReadLine();
                    if (line == null) break;
                }
                catch
                {
                    break;
                }

                if (line.StartsWith("STATUS "))
                {
                    var msg = line.Substring("STATUS ".Length);
                    BeginInvoke(new Action(() => UpdateStatus(msg)));
                }
                else if (line.StartsWith("FRAME "))
                {
                    var b64 = line.Substring("FRAME ".Length);
                    try
                    {
                        var jpg = Convert.FromBase64String(b64);
                        using var ms = new MemoryStream(jpg);
                        using var img = Image.FromStream(ms);

                        var clone = new Bitmap(img);

                        BeginInvoke(new Action(() =>
                        {
                            var old = pic.Image;
                            pic.Image = clone;
                            old?.Dispose();
                        }));
                    }
                    catch
                    {
                        // ignore frame decode errors
                    }
                }
            }

            BeginInvoke(new Action(() => UpdateStatus("Disconnected")));
        }

        private void UpdateStatus(string s)
        {
            lblStatus.Text = "Status: " + s;
        }
    }
}
