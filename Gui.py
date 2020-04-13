import cv2


class Gui:
    """Provides a managed environment for tuning OpenCV parameters."""

    def __init__(self, key_handler=None):
        self.widgets = []
        self.state = {"gui_paused": False, "gui_quitting": False}
        self.key_handler = key_handler

    def __del__(self):
        self.hide()

    def show(self):
        """Shows the window."""

        # create the widgets
        for widget in self.widgets:
            widget.show()

        self.state["gui_quitting"] = False

        while not self.state["gui_quitting"]:
            imgs = []

            if not self.state["gui_paused"]:
                for widget in self.widgets:
                    widget.step(imgs, self.state)
                    imgs.append(widget.img)

            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or key == ord('Q'):
                print("GUI is quitting")
                self.state["gui_quitting"] = True
            elif key == ord('p') or key == ord('P'):
                print("GUI is paused")
                self.state["gui_paused"] = not self.state["gui_paused"]
            else:
                self.key_handler(key, imgs, self.state)

        self.hide()

    def hide(self):
        """Hides the window."""
        for widget in self.widgets:
            widget.hide()


class Param:

    def __init__(self, name, minimum, maximum, default):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.default = default


class Widget:

    def __init__(self, name, function, show_image=True, show_window=True, display_function=None):
        self.name = name
        self.params = []
        self.function = function
        self.img = None
        self.display_function = display_function
        self.show_window = show_window
        self.show_img = show_image

    def __del__(self):
        self.hide()

    def __nop(self, _):
        """Does nothing."""

    def show(self):
        """Shows the window."""
        assert self.show_window or (
            not self.show_window and len(self.params) == 0)

        if self.show_window:
            cv2.namedWindow(self.name)
            for param in self.params:
                cv2.createTrackbar(param.name, self.name,
                                   param.default, param.maximum, self.__nop)
                cv2.setTrackbarMin(param.name, self.name, param.minimum)

    def hide(self):
        """Hides the window."""
        cv2.destroyWindow(self.name)

    def step(self, imgs, state):
        """Executes the step, updating its image and window."""
        latest_img = None
        if len(imgs) > 0:
            latest_img = imgs[-1]

        evaluated_params = {}
        for param in self.params:
            value = cv2.getTrackbarPos(param.name, self.name)
            evaluated_params[param.name] = value

        shown_img = self.img = self.function(
            latest_img, evaluated_params, imgs, state)

        if self.display_function is not None:
            shown_img = self.display_function(
                self.img, evaluated_params, imgs + [self.img], state)

        if self.show_img and self.show_window:
            cv2.imshow(self.name, shown_img)
