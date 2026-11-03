import sys
from PySide6.QtWidgets import QApplication
from ui import MainWindow

if __name__ == "__main__":
    app = QApplication()
    w = MainWindow(app.primaryScreen().size())
    w.show()
    sys.exit(app.exec())