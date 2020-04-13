import cv2
import os

DIR_NAME = "./die_images/"

WIN_NAME = "Classify Image"


def classify_image(filename, number):
    if number < 0 or number > 6:
        raise ValueError()

    new_filename = os.path.join(os.path.dirname(filename), str(
        number) + "_" + os.path.basename(filename))
    os.rename(filename, new_filename)

    print("Renamed", os.path.basename(filename),
          "to", os.path.basename(new_filename))


def print_stats():
    unclassified = 0
    nums = [0, 0, 0, 0, 0, 0, 0]
    for filename in os.listdir(DIR_NAME):
        parts = filename.split("_")
        if len(parts) == 2:
            unclassified += 1
        elif len(parts) == 3:
            nums[int(parts[0])] += 1
    print("There are", unclassified, "unclassifed images")
    print("There are:")
    for i, num in enumerate(nums):
        print("\t{:3} {}s".format(num, i))


def main():
    filenames = []
    for filename in os.listdir(DIR_NAME):
        parts = filename.split("_")
        if len(parts) == 2:
            filenames.append(os.path.join(DIR_NAME, filename))

    print_stats()

    if len(filenames) > 0:

        print()
        print("Keys:")
        print("\t1 - 6: Classify image as a number")
        print("\td: Delete image")
        print("\ts: Skip image")
        print("\tq: Quit")

        running = True
        while running and len(filenames) > 0:
            filename = filenames.pop()
            img = cv2.imread(filename)
            cv2.namedWindow(filename)
            cv2.moveWindow(filename, 100, 100)
            cv2.imshow(filename, img)
            number = None

            while True:
                key = cv2.waitKey(1) & 0xff
                if key == ord('q') or key == ord('Q'):
                    running = False
                    print("Quitting")
                    break
                elif key == ord('d') or key == ord('D'):
                    print("Deleting", os.path.basename(filename))
                    os.unlink(filename)
                    break
                elif key == ord('s') or key == ord('S'):
                    print("Skipping", os.path.basename(filename))
                    break
                elif (key - ord('0')) >= 0 and (key - ord('0') <= 6):
                    number = key - ord('0')
                    break

            if number is not None:
                classify_image(filename, number)
            cv2.destroyWindow(filename)

        cv2.destroyAllWindows()
        print_stats()


if __name__ == "__main__":
    main()
