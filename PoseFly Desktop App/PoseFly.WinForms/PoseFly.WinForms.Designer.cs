namespace PoseFly.WinForms
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        // ---- Controls ----
        private System.Windows.Forms.TableLayoutPanel root;
        private System.Windows.Forms.GroupBox grpInference;
        private System.Windows.Forms.FlowLayoutPanel flowInference;

        private System.Windows.Forms.CheckBox chkDrone;
        private System.Windows.Forms.CheckBox chkAngle;
        private System.Windows.Forms.CheckBox chkDistance;
        private System.Windows.Forms.CheckBox chkLed;

        private System.Windows.Forms.GroupBox grpSettings;
        private System.Windows.Forms.TableLayoutPanel gridSettings;

        private System.Windows.Forms.CheckBox chkSave;
        private System.Windows.Forms.Label lblCameraIndex;
        private System.Windows.Forms.NumericUpDown numCamera;

        private System.Windows.Forms.Label lblFps;
        private System.Windows.Forms.TextBox txtFps;

        // ✅ Rolling shutter controls
        private System.Windows.Forms.Label lblIso;
        private System.Windows.Forms.NumericUpDown numIso;
        private System.Windows.Forms.Label lblShutterHz;
        private System.Windows.Forms.NumericUpDown numShutterHz;

        private System.Windows.Forms.Label lblOutputPath;
        private System.Windows.Forms.TextBox txtOutput;
        private System.Windows.Forms.Button btnBrowse;

        private System.Windows.Forms.GroupBox grpControls;
        private System.Windows.Forms.FlowLayoutPanel flowControls;

        private System.Windows.Forms.Button btnStart;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.Button btnQuit;
        private System.Windows.Forms.Label lblStatus;

        private System.Windows.Forms.GroupBox grpLiveView;
        private System.Windows.Forms.PictureBox pic;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
                components.Dispose();

            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            root = new TableLayoutPanel();
            grpInference = new GroupBox();
            flowInference = new FlowLayoutPanel();
            chkDrone = new CheckBox();
            chkAngle = new CheckBox();
            chkDistance = new CheckBox();
            chkLed = new CheckBox();
            grpSettings = new GroupBox();
            gridSettings = new TableLayoutPanel();
            chkSave = new CheckBox();
            lblCameraIndex = new Label();
            numCamera = new NumericUpDown();
            lblFps = new Label();
            txtFps = new TextBox();

            // ✅ rolling shutter init
            lblIso = new Label();
            numIso = new NumericUpDown();
            lblShutterHz = new Label();
            numShutterHz = new NumericUpDown();

            lblOutputPath = new Label();
            txtOutput = new TextBox();
            btnBrowse = new Button();
            grpControls = new GroupBox();
            flowControls = new FlowLayoutPanel();
            btnStart = new Button();
            btnStop = new Button();
            btnQuit = new Button();
            lblStatus = new Label();
            grpLiveView = new GroupBox();
            pic = new PictureBox();

            root.SuspendLayout();
            grpInference.SuspendLayout();
            flowInference.SuspendLayout();
            grpSettings.SuspendLayout();
            gridSettings.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)numCamera).BeginInit();

            // ✅ BeginInit for new numeric controls
            ((System.ComponentModel.ISupportInitialize)numIso).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numShutterHz).BeginInit();

            grpControls.SuspendLayout();
            flowControls.SuspendLayout();
            grpLiveView.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)pic).BeginInit();
            SuspendLayout();

            // 
            // root
            // 
            root.ColumnCount = 1;
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            root.Controls.Add(grpInference, 0, 0);
            root.Controls.Add(grpSettings, 0, 1);
            root.Controls.Add(grpControls, 0, 2);
            root.Controls.Add(grpLiveView, 0, 3);
            root.Dock = DockStyle.Fill;
            root.Location = new Point(0, 0);
            root.Name = "root";
            root.Padding = new Padding(12);
            root.RowCount = 4;
            root.RowStyles.Add(new RowStyle());
            root.RowStyles.Add(new RowStyle());
            root.RowStyles.Add(new RowStyle());
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            root.Size = new Size(920, 820);
            root.TabIndex = 0;

            // 
            // grpInference
            // 
            grpInference.Controls.Add(flowInference);
            grpInference.Dock = DockStyle.Top;
            grpInference.Location = new Point(15, 15);
            grpInference.Name = "grpInference";
            grpInference.Size = new Size(890, 120);
            grpInference.TabIndex = 0;
            grpInference.TabStop = false;
            grpInference.Text = "Inference";

            // 
            // flowInference
            // 
            flowInference.Controls.Add(chkDrone);
            flowInference.Controls.Add(chkAngle);
            flowInference.Controls.Add(chkDistance);
            flowInference.Controls.Add(chkLed);
            flowInference.Dock = DockStyle.Fill;
            flowInference.FlowDirection = FlowDirection.TopDown;
            flowInference.Location = new Point(3, 19);
            flowInference.Name = "flowInference";
            flowInference.Size = new Size(884, 98);
            flowInference.TabIndex = 0;
            flowInference.WrapContents = false;

            // 
            // chkDrone
            // 
            chkDrone.AutoSize = true;
            chkDrone.Checked = true;
            chkDrone.CheckState = CheckState.Checked;
            chkDrone.Location = new Point(3, 3);
            chkDrone.Name = "chkDrone";
            chkDrone.Size = new Size(112, 19);
            chkDrone.TabIndex = 0;
            chkDrone.Text = "Drone Detection";

            // 
            // chkAngle
            // 
            chkAngle.AutoSize = true;
            chkAngle.Checked = true;
            chkAngle.CheckState = CheckState.Checked;
            chkAngle.Location = new Point(3, 28);
            chkAngle.Name = "chkAngle";
            chkAngle.Size = new Size(109, 19);
            chkAngle.TabIndex = 1;
            chkAngle.Text = "Angle Inference";

            // 
            // chkDistance
            // 
            chkDistance.AutoSize = true;
            chkDistance.Checked = true;
            chkDistance.CheckState = CheckState.Checked;
            chkDistance.Location = new Point(3, 53);
            chkDistance.Name = "chkDistance";
            chkDistance.Size = new Size(123, 19);
            chkDistance.TabIndex = 2;
            chkDistance.Text = "Distance Inference";

            // 
            // chkLed
            // 
            chkLed.AutoSize = true;
            chkLed.Checked = true;
            chkLed.CheckState = CheckState.Checked;
            chkLed.Location = new Point(3, 78);
            chkLed.Name = "chkLed";
            chkLed.Size = new Size(112, 19);
            chkLed.TabIndex = 3;
            chkLed.Text = "LED ID Inference";

            // 
            // grpSettings
            // 
            grpSettings.Controls.Add(gridSettings);
            grpSettings.Dock = DockStyle.Top;
            grpSettings.Location = new Point(15, 141);
            grpSettings.Name = "grpSettings";
            grpSettings.Size = new Size(890, 190); // ✅ slightly taller for extra row
            grpSettings.TabIndex = 1;
            grpSettings.TabStop = false;
            grpSettings.Text = "Settings";

            // 
            // gridSettings
            // 
            gridSettings.ColumnCount = 5;
            gridSettings.ColumnStyles.Add(new ColumnStyle());
            gridSettings.ColumnStyles.Add(new ColumnStyle());
            gridSettings.ColumnStyles.Add(new ColumnStyle());
            gridSettings.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            gridSettings.ColumnStyles.Add(new ColumnStyle());

            gridSettings.Controls.Add(chkSave, 0, 0);

            gridSettings.Controls.Add(lblCameraIndex, 0, 1);
            gridSettings.Controls.Add(numCamera, 1, 1);
            gridSettings.Controls.Add(lblFps, 2, 1);
            gridSettings.Controls.Add(txtFps, 3, 1);

            // ✅ new row for ISO / Shutter Hz
            gridSettings.Controls.Add(lblIso, 0, 2);
            gridSettings.Controls.Add(numIso, 1, 2);
            gridSettings.Controls.Add(lblShutterHz, 2, 2);
            gridSettings.Controls.Add(numShutterHz, 3, 2);

            gridSettings.Controls.Add(lblOutputPath, 0, 3);
            gridSettings.Controls.Add(txtOutput, 1, 3);
            gridSettings.Controls.Add(btnBrowse, 4, 3);

            gridSettings.Dock = DockStyle.Fill;
            gridSettings.Location = new Point(3, 19);
            gridSettings.Name = "gridSettings";

            // ✅ was 3, now 4
            gridSettings.RowCount = 4;
            gridSettings.RowStyles.Add(new RowStyle());
            gridSettings.RowStyles.Add(new RowStyle());
            gridSettings.RowStyles.Add(new RowStyle());
            gridSettings.RowStyles.Add(new RowStyle());

            gridSettings.Size = new Size(884, 168);
            gridSettings.TabIndex = 0;

            // 
            // chkSave
            // 
            chkSave.AutoSize = true;
            chkSave.Checked = true;
            chkSave.CheckState = CheckState.Checked;
            chkSave.Location = new Point(3, 3);
            chkSave.Name = "chkSave";
            chkSave.Size = new Size(83, 19);
            chkSave.TabIndex = 0;
            chkSave.Text = "Save Video";

            // 
            // lblCameraIndex
            // 
            lblCameraIndex.AutoSize = true;
            lblCameraIndex.Location = new Point(3, 25);
            lblCameraIndex.Name = "lblCameraIndex";
            lblCameraIndex.Padding = new Padding(0, 8, 0, 0);
            lblCameraIndex.Size = new Size(82, 23);
            lblCameraIndex.TabIndex = 1;
            lblCameraIndex.Text = "Camera Index:";

            // 
            // numCamera
            // 
            numCamera.Location = new Point(92, 28);
            numCamera.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
            numCamera.Name = "numCamera";
            numCamera.Size = new Size(80, 23);
            numCamera.TabIndex = 2;

            // 
            // lblFps
            // 
            lblFps.AutoSize = true;
            lblFps.Location = new Point(178, 25);
            lblFps.Name = "lblFps";
            lblFps.Padding = new Padding(0, 8, 0, 0);
            lblFps.Size = new Size(29, 23);
            lblFps.TabIndex = 3;
            lblFps.Text = "FPS:";

            // 
            // txtFps
            // 
            txtFps.Location = new Point(213, 28);
            txtFps.Name = "txtFps";
            txtFps.Size = new Size(80, 23);
            txtFps.TabIndex = 4;
            txtFps.Text = "10.0";

            // 
            // lblIso
            // 
            lblIso.AutoSize = true;
            lblIso.Location = new Point(3, 54);
            lblIso.Name = "lblIso";
            lblIso.Padding = new Padding(0, 8, 0, 0);
            lblIso.Size = new Size(27, 23);
            lblIso.TabIndex = 5;
            lblIso.Text = "ISO:";

            // 
            // numIso
            // 
            numIso.Location = new Point(92, 57);
            numIso.Minimum = new decimal(new int[] { 50, 0, 0, 0 });
            numIso.Maximum = new decimal(new int[] { 6400, 0, 0, 0 });
            numIso.Increment = new decimal(new int[] { 50, 0, 0, 0 });
            numIso.Name = "numIso";
            numIso.Size = new Size(80, 23);
            numIso.TabIndex = 6;
            numIso.Value = new decimal(new int[] { 100, 0, 0, 0 });

            // 
            // lblShutterHz
            // 
            lblShutterHz.AutoSize = true;
            lblShutterHz.Location = new Point(178, 54);
            lblShutterHz.Name = "lblShutterHz";
            lblShutterHz.Padding = new Padding(0, 8, 0, 0);
            lblShutterHz.Size = new Size(69, 23);
            lblShutterHz.TabIndex = 7;
            lblShutterHz.Text = "Shutter Hz:";

            // 
            // numShutterHz
            // 
            numShutterHz.Location = new Point(253, 57);
            numShutterHz.Minimum = new decimal(new int[] { 5, 0, 0, 0 });
            numShutterHz.Maximum = new decimal(new int[] { 6000, 0, 0, 0 });
            numShutterHz.Increment = new decimal(new int[] { 5, 0, 0, 0 });
            numShutterHz.Name = "numShutterHz";
            numShutterHz.Size = new Size(90, 23);
            numShutterHz.TabIndex = 8;
            numShutterHz.Value = new decimal(new int[] { 1000, 0, 0, 0 });

            // 
            // lblOutputPath
            // 
            lblOutputPath.AutoSize = true;
            lblOutputPath.Location = new Point(3, 83);
            lblOutputPath.Name = "lblOutputPath";
            lblOutputPath.Padding = new Padding(0, 8, 0, 0);
            lblOutputPath.Size = new Size(75, 23);
            lblOutputPath.TabIndex = 9;
            lblOutputPath.Text = "Output Path:";

            // 
            // txtOutput
            // 
            txtOutput.Anchor = AnchorStyles.Left | AnchorStyles.Right;
            gridSettings.SetColumnSpan(txtOutput, 3);
            txtOutput.Location = new Point(92, 86);
            txtOutput.Name = "txtOutput";
            txtOutput.Size = new Size(693, 23);
            txtOutput.TabIndex = 10;
            txtOutput.Text = "results\\full_pipeline_results\\posefly_results2.mp4";

            // 
            // btnBrowse
            // 
            btnBrowse.Location = new Point(791, 86);
            btnBrowse.Name = "btnBrowse";
            btnBrowse.Size = new Size(90, 23);
            btnBrowse.TabIndex = 11;
            btnBrowse.Text = "Browse";
            btnBrowse.Click += BtnBrowse_Click;

            // 
            // grpControls
            // 
            grpControls.Controls.Add(flowControls);
            grpControls.Dock = DockStyle.Top;
            grpControls.Location = new Point(15, 337);
            grpControls.Name = "grpControls";
            grpControls.Size = new Size(890, 90);
            grpControls.TabIndex = 2;
            grpControls.TabStop = false;
            grpControls.Text = "Controls";

            // 
            // flowControls
            // 
            flowControls.Controls.Add(btnStart);
            flowControls.Controls.Add(btnStop);
            flowControls.Controls.Add(btnQuit);
            flowControls.Controls.Add(lblStatus);
            flowControls.Dock = DockStyle.Fill;
            flowControls.Location = new Point(3, 19);
            flowControls.Name = "flowControls";
            flowControls.Size = new Size(884, 68);
            flowControls.TabIndex = 0;
            flowControls.WrapContents = false;

            // 
            // btnStart
            // 
            btnStart.Location = new Point(3, 3);
            btnStart.Name = "btnStart";
            btnStart.Size = new Size(100, 23);
            btnStart.TabIndex = 0;
            btnStart.Text = "Start";
            btnStart.Click += BtnStart_Click;

            // 
            // btnStop
            // 
            btnStop.Location = new Point(109, 3);
            btnStop.Name = "btnStop";
            btnStop.Size = new Size(100, 23);
            btnStop.TabIndex = 1;
            btnStop.Text = "Stop";
            btnStop.Click += BtnStop_Click;

            // 
            // btnQuit
            // 
            btnQuit.Location = new Point(215, 3);
            btnQuit.Name = "btnQuit";
            btnQuit.Size = new Size(100, 23);
            btnQuit.TabIndex = 2;
            btnQuit.Text = "Quit";
            btnQuit.Click += BtnQuit_Click;

            // 
            // lblStatus
            // 
            lblStatus.AutoSize = true;
            lblStatus.Location = new Point(321, 0);
            lblStatus.Name = "lblStatus";
            lblStatus.Padding = new Padding(10, 10, 0, 0);
            lblStatus.Size = new Size(74, 25);
            lblStatus.TabIndex = 3;
            lblStatus.Text = "Status: Idle";

            // 
            // grpLiveView
            // 
            grpLiveView.Controls.Add(pic);
            grpLiveView.Dock = DockStyle.Fill;
            grpLiveView.Location = new Point(15, 433);
            grpLiveView.Name = "grpLiveView";
            grpLiveView.Size = new Size(890, 372);
            grpLiveView.TabIndex = 3;
            grpLiveView.TabStop = false;
            grpLiveView.Text = "Live View";

            // 
            // pic
            // 
            pic.BorderStyle = BorderStyle.FixedSingle;
            pic.Dock = DockStyle.Fill;
            pic.Location = new Point(3, 19);
            pic.Name = "pic";
            pic.Size = new Size(884, 350);
            pic.SizeMode = PictureBoxSizeMode.Zoom;
            pic.TabIndex = 0;
            pic.TabStop = false;

            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.Control;
            ClientSize = new Size(920, 820);
            Controls.Add(root);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "Form1";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "PoseFly - AI-enhanced Drone-to-Drone Communication System";
            FormClosing += Form1_FormClosing;

            root.ResumeLayout(false);
            grpInference.ResumeLayout(false);
            flowInference.ResumeLayout(false);
            flowInference.PerformLayout();
            grpSettings.ResumeLayout(false);
            gridSettings.ResumeLayout(false);
            gridSettings.PerformLayout();

            ((System.ComponentModel.ISupportInitialize)numCamera).EndInit();
            ((System.ComponentModel.ISupportInitialize)numIso).EndInit();
            ((System.ComponentModel.ISupportInitialize)numShutterHz).EndInit();

            grpControls.ResumeLayout(false);
            flowControls.ResumeLayout(false);
            flowControls.PerformLayout();
            grpLiveView.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)pic).EndInit();
            ResumeLayout(false);
        }

        #endregion
    }
}