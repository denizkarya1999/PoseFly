# main.py
from backend import PipelineBackend
from ui import PipelineUI


def main():
    backend = PipelineBackend()
    app = PipelineUI(backend)
    app.mainloop()


if __name__ == "__main__":
    main()