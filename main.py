import cv2
import numpy as np
from pydantic.v1.main import validate_custom_root_type

import variables
from threading import Thread
from imgui_bundle import implot
import eye_model
from imgui_bundle import immapp, imgui
from dataset import *
from utils import *
from scipy.interpolate import interp1d

TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_vals = np.linspace(0, 1, 1000)
y_vals = blink_curve(x_vals)
inv_blink_curve = interp1d(y_vals, x_vals, fill_value="extrapolate", kind="linear")
inv_y_vals = inv_blink_curve(y_vals)

time_graph: list[float] = []
graph: list[float] = []
inv_graph: list[float] = []

time: float = 0

eye_smoother = OutputSmoother()
blink_smoother = OutputSmoother(t=2)
focus_smoother = OutputSmoother(alpha=1)
dilation_smoother = OutputSmoother(alpha=.8, t=3, use_diff=True)

def draw_blink_bar(img, blink_value, max_height=50, bar_width=15, bar_x_offset=0, color=(0, 0, 255)):
    h, w = img.shape[:2]
    bar_height = int(max_height * blink_value)
    x1 = 5 + bar_x_offset * bar_width
    y1 = h - 5
    x2 = x1 + bar_width
    y2 = y1 - bar_height
    cv2.rectangle(img, (x1, y1 - max_height), (x2, y1), (50, 50, 50), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img

def get_pupil_dilation(img, percentile=2, name="left"):
    h, w = img.shape[:2]
    if name == "left":
        img = img[h//2:, :w//3 * 2].copy()
    else:
        img = img[h//2:, w//3:].copy()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce small noise
    blur = cv2.GaussianBlur(gray, (5,5), 5)

    blur = cv2.convertScaleAbs(blur, alpha=5, beta=2).astype(np.uint8)

    blur = ( (((blur - np.min(blur)) / 255) * 255)).astype(np.uint8)

    # Take the darkest X% of pixels
    thresh_val = np.percentile(blur, percentile)
    mask = (blur <= thresh_val).astype(np.uint8) * 255

    #Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # Biggest dark blob = pupil
    pupil_contour = max(contours, key=cv2.contourArea)

    # Fit circle
    (x, y), radius = cv2.minEnclosingCircle(pupil_contour)
    radius = int(radius)

    #radius = np.clip(remap(radius, 22, 24, 0, 1), 0, 1)

    cv2.circle(blur, (int(x), int(y)), int(radius), (255, 255, 255), 2)

    cv2.imshow(f"{name}blur", blur)
    cv2.imshow(f"{name} mask", mask)

    return (int(x), int(y)), radius


def live_preview(model_left, model_right):
    global time
    model_left.eval()
    model_right.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        eye_smoother.alpha = variables.eye_smoothness
        blink_smoother.alpha = variables.blink_smoothness
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        mid = w // 2

        left_eye = frame[:, :mid].copy()
        right_eye = frame[:, mid:].copy()

        def to_tensor(bgr_img):
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            return eye_model.runtime_transform(pil).unsqueeze(0).to(TARGET_DEVICE)

        lt = to_tensor(left_eye)
        rt = to_tensor(right_eye)

        with torch.no_grad():
            gaze_pred_l, eyelid_pred_l, dilation_pred_l = model_left(lt)
            gaze_pred_r, eyelid_pred_r, dilation_pred_r = model_right(rt)
            gaze_pred_l, gaze_pred_r = gaze_pred_l[0].cpu().numpy(), gaze_pred_r[0].cpu().numpy()
            eyelid_pred_l, eyelid_pred_r = eyelid_pred_l[0].cpu().numpy()[0], eyelid_pred_r[0].cpu().numpy()[0]

        lx, ly, rx, ry = eye_smoother.update([gaze_pred_l[0], gaze_pred_l[1], gaze_pred_r[0], gaze_pred_r[1]])
        eyelid_pred_l, eyelid_pred_r = blink_smoother.update([eyelid_pred_l, eyelid_pred_r])

        if np.abs(eyelid_pred_l - eyelid_pred_r) < 1:
            min_lid = min(eyelid_pred_l, eyelid_pred_r)
            eyelid_pred_l = min_lid
            eyelid_pred_r = min_lid

        l_pupil_center, l_pupil_radius = get_pupil_dilation(left_eye)
        r_pupil_center, r_pupil_radius = get_pupil_dilation(right_eye, name="right")

        pupil_dilation = dilation_smoother.update([(l_pupil_radius + r_pupil_radius) / 2.0])[0]
        norm_pupil_dilation = np.clip(remap(pupil_dilation, 20, 33, 0, 1)**.8, 0, 1)

        lx = ((lx - 0.5) * variables.eye_x_scale) + 0.5
        rx = ((rx - 0.5) * variables.eye_x_scale) + 0.5
        ly = ((ly - 0.5) * variables.eye_y_scale) + 0.5
        ry = ((ry - 0.5) * variables.eye_y_scale) + 0.5

        send_eye_osc((lx - .5) * 2, (1 - ly - .5) * 2, eyelid_pred_l, (rx - .5) * 2, (1 - ry - .5) * 2, eyelid_pred_r, norm_pupil_dilation)

        cv2.circle(left_eye, (l_pupil_center[0], l_pupil_center[1] + h // 2), l_pupil_radius, (255, 255, 255), 2)
        cv2.circle(right_eye, (r_pupil_center[0] +  + w // 3 // 2, r_pupil_center[1] + h // 2), r_pupil_radius, (255, 255, 255), 2)

        cv2.circle(left_eye, (int(lx * left_eye.shape[1]), int(ly * left_eye.shape[0])), 8, (0, 255, 0), -1)
        cv2.circle(right_eye, (int(rx * right_eye.shape[1]), int(ry * right_eye.shape[0])), 8, (0, 255, 0), -1)

        draw_blink_bar(left_eye, eyelid_pred_l, 120)
        draw_blink_bar(right_eye, eyelid_pred_r, 120)

        draw_blink_bar(left_eye, norm_pupil_dilation, 120, color=(0, 255, 0), bar_x_offset=2)
        draw_blink_bar(right_eye, norm_pupil_dilation, 120, color=(0, 255, 0), bar_x_offset=2)

        time_graph.append(time)
        focus_s = focus_smoother.update([np.clip(lx - rx, 0, 1)])[0]
        graph.append(focus_s)
        time += .01

        cv2.imshow("Left Eye", left_eye)
        cv2.imshow("Right Eye", right_eye)


        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break
        elif not variables.is_eye_tracking_running:
            break

    cap.release()
    cv2.destroyAllWindows()

def train_new_models():
    variables.EPOCHS = 50
    eye_model.train_both_models(eye_model.left_eye_model, eye_model.right_eye_model, variables.x_max_abs, variables.y_max_abs, 1, 0, 0)
    variables.EPOCHS = 5
    eye_model.train_both_models(eye_model.left_eye_model, eye_model.right_eye_model, variables.x_max_abs, variables.y_max_abs, .1, 1, 0)
    # variables.EPOCHS = 20
    # eye_model.train_both_models(eye_model.left_eye_model, eye_model.right_eye_model, variables.x_max_abs, variables.y_max_abs, 0, 1, 0)
    eye_model.save_eye_models()

def draw_imgui():
    imgui.style_colors_classic()

    if imgui.begin_tab_bar("MyTabBar"):
        if imgui.begin_tab_item("Main")[0]:
            imgui.separator_text("Main")

            if variables.found_models:
                imgui.bullet_text("Found Left Model")
                imgui.bullet_text("Found Right Model")
                if imgui.button("Start/Stop ET"):
                    variables.is_eye_tracking_running = not variables.is_eye_tracking_running

                    if variables.is_eye_tracking_running:
                        eye_model.left_eye_model = eye_model.EyeCNN().to(TARGET_DEVICE)
                        eye_model.right_eye_model = eye_model.EyeCNN().to(TARGET_DEVICE)

                        eye_model.left_eye_model.load_state_dict(torch.load(variables.MODEL_LEFT_PATH, map_location=TARGET_DEVICE))
                        eye_model.right_eye_model.load_state_dict(torch.load(variables.MODEL_RIGHT_PATH, map_location=TARGET_DEVICE))

                        variables.eye_tracking_thread = Thread(target=live_preview, args=(eye_model.left_eye_model, eye_model.right_eye_model), daemon=True)
                        variables.eye_tracking_thread.start()
                    else:
                        variables.eye_tracking_thread.join()
            else:
                imgui.bullet_text("No Models Found")

            if implot.begin_plot("Blinking"):
                implot.plot_line("Raw", np.array(time_graph), np.array(graph))
                implot.end_plot()
            if implot.begin_plot("graphs"):
                implot.plot_line("norm", x_vals, y_vals)
                implot.plot_line("inv", y_vals, np.array(inv_y_vals))
                implot.plot_line("out", x_vals, np.array(inv_y_vals))
                implot.end_plot()
            imgui.end_tab_item()

        if imgui.begin_tab_item("Settings")[0]:
            imgui.separator_text("Settings")

            variables.eye_x_scale = imgui.slider_float("X Scale", variables.eye_x_scale, 0, 2, "%.1F")[1]
            variables.eye_y_scale = imgui.slider_float("Y Scale", variables.eye_y_scale, 0, 2, "%.1F")[1]
            variables.eye_smoothness = imgui.slider_float("Gaze Smoothness", variables.eye_smoothness, 0, 1, "%.1F")[1]
            variables.blink_smoothness = imgui.slider_float("Blink Smoothness", variables.blink_smoothness, 0, 1, "%.1F")[1]

            imgui.end_tab_item()

        if imgui.begin_tab_item("Model")[0]:
            imgui.separator_text("Model")
            if not variables.found_models:
                imgui.text("No models found")
                if imgui.button("Try Find Models"):
                    variables.found_models = os.path.exists(
                        variables.MODEL_LEFT_PATH) and os.path.exists(variables.MODEL_RIGHT_PATH)

            if imgui.button("Train New Models"):
                eye_model.left_eye_model = eye_model.EyeCNN().to(TARGET_DEVICE)
                eye_model.right_eye_model = eye_model.EyeCNN().to(TARGET_DEVICE)
                Thread(target=train_new_models).start()

            if implot.begin_plot("Training"):
                implot.setup_axes_limits(0, variables.EPOCHS, 0, .1)
                implot.plot_line("L Training", np.array(eye_model.left_eye_graph))
                implot.plot_line("L Val", np.array(eye_model.left_eye_val_graph))
                implot.plot_line("R Training", np.array(eye_model.right_eye_graph))
                implot.plot_line("R Val", np.array(eye_model.right_eye_val_graph))
                implot.end_plot()

            imgui.end_tab_item()

        imgui.end_tab_bar()

def run_imgui():
    immapp.run(gui_function=draw_imgui,
               with_implot=True,
               with_markdown=True,
               window_size_auto=False,
               window_title="N2 ET",
               fps_idle=60)

if __name__ == "__main__":
    run_imgui()