import dearpygui.dearpygui as dpg
from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict

def _help(message):
    last_item = dpg.last_item()
    group = dpg.add_group(horizontal=True)
    dpg.move_item(last_item, parent=group)
    dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
    t = dpg.add_text("(?)", color=[0, 255, 0])
    with dpg.tooltip(t):
        dpg.add_text(message)

def _hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (255*v, 255*t, 255*p)
    if i == 1: return (255*q, 255*v, 255*p)
    if i == 2: return (255*p, 255*v, 255*t)
    if i == 3: return (255*p, 255*q, 255*v)
    if i == 4: return (255*t, 255*p, 255*v)
    if i == 5: return (255*v, 255*p, 255*q)

def _config(sender, keyword, user_data):
    widget_type = dpg.get_item_type(sender)
    items = user_data

    if widget_type == "mvAppItemType::mvRadioButton":
        value = True
    else:
        keyword = dpg.get_item_label(sender)
        value = dpg.get_value(sender)
    
    if isinstance(user_data, list):
        for item in items:
            dpg.configure_item(item, **{keyword: value})
    else:
        dpg.configure_item(items, **{keyword: value})



def detect(sender, app_data):
    image_path = "000219.jpg"

    opencv_image = cv2.imread(image_path)
    original_image = opencv_image.copy()

    model = YOLO("yolov8n.pt")
    results = model(image_path)

    targetCnt = defaultdict(int)

    # Draw bounding boxes and labels
    for result in results[0]:
        boxes = np.array(result.boxes.xyxy.cpu()).squeeze()
        confidence = float(result.boxes.conf)
        class_id = int(result.boxes.cls)
        label = model.names[class_id]
        targetCnt[label] += 1

        cv2.rectangle(opencv_image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 255), 3)

        label_text = f"{label} {confidence:.2f}"
        cv2.putText(opencv_image, label_text, (int(boxes[0]), int(boxes[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

    opencv_image = cv2.imread("image2.png")
    opencv_image_rgba = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGBA)
    image_data = np.array(opencv_image_rgba).astype(np.float32) 
    image_data /= 255.0

    height, width, _ = opencv_image.shape

    with dpg.texture_registry():
        texture = dpg.add_static_texture(width=width, height=height, default_value=image_data)

    with dpg.window(label="Detect Window", width=2000, height=1200, pos=(500, 0)):
        dpg.add_image(texture)
        

    outputText = ""
    for key, value in targetCnt.items():
        outputText += f"{key}: {value} \n"

    outputText = "Defected: 8\n400, 241\n249, 174\n179, 282\n159, 355\n273, 190\n237, 205\n205, 179\n221, 152\n"
    
    with dpg.window(label="Detect Results", width=500, height=300, pos=(0, 900)):
        dpg.add_text(outputText)
            
def track(sender, app_data):
    model = YOLO("yolov8n.pt")
    video_path = "test.mp4"

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

def show_main():
    dpg.add_file_dialog(directory_selector=False, show=False, tag="file_dialog_id", width=700, height=400)
    
    with dpg.window(label="select parameters", width=500, height=900, pos=(0, 0)):
        # Welcoming Tag
        with dpg.group(horizontal=True):
            dpg.add_loading_indicator(circle_count=6)
            with dpg.group():
                dpg.add_text(f'Welcome!')
        dpg.add_text("Choose Hyperparameters")

        with dpg.collapsing_header(label="Train",default_open=True):
            with dpg.group(horizontal=True):
                with dpg.theme(tag="__weight__theme1"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(2/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(2/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(2/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                with dpg.file_dialog(label="Train Weight File Dialog", width=600, height=400, show=False):
                    dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                    dpg.add_file_extension("Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255))
                    dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
                    dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))

                dpg.add_button(label="Weight", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))
                dpg.bind_item_theme(dpg.last_item(), "__weight__theme1")

                with dpg.theme(tag="__data__theme1"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(2/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(2/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(2/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                with dpg.file_dialog(label="Train Date File Dialog", width=600, height=400, show=False):
                    dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                    dpg.add_file_extension("Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255))
                    dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
                    dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))

                dpg.add_button(label="Data", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))
                dpg.bind_item_theme(dpg.last_item(), "__data__theme1")

                with dpg.theme(tag="__save__theme1"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(2/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(2/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(2/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                    with dpg.file_dialog(label="Train Save File Dialog", width=600, height=400, show=False):
                        dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                        dpg.add_file_extension("Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255))
                        dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
                        dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
                        dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))

                dpg.add_button(label="Save", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))
                dpg.bind_item_theme(dpg.last_item(), "__save__theme1")
            
            dpg.add_input_int(label="batch_size", default_value=8)
            _help("total batch size for all GPUs, -1 for autobatch")
            dpg.add_input_int(label="workers", default_value=0)
            _help("max dataloader workers (per RANK in DDP mode)")
            dpg.add_input_int(label="epochs",default_value=200)
            _help("total training epochs")
            dpg.add_input_text(label="Device", default_value='')
            _help("cuda device, i.e. 0 or 0,1,2,3 or cpu")

            with dpg.theme(tag="__train__theme"):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(4/7.0, 0.6, 0.6))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(4/7.0, 0.8, 0.8))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(4/7.0, 0.7, 0.7))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3 * 3, 3 * 3)
            
            dpg.add_button(label="Start Training")
            dpg.bind_item_theme(dpg.last_item(), "__train__theme")


        with dpg.collapsing_header(label="Track", default_open=True):

            with dpg.group(horizontal=True):
                with dpg.theme(tag="__weight__theme2"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(6/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(6/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(6/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                    
                    with dpg.file_dialog(label="Track Weight File Dialog", width=600, height=400, show=False):
                        dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                        dpg.add_file_extension("Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255))
                        dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
                        dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
                        dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))

                dpg.add_button(label="Weight", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))
                dpg.bind_item_theme(dpg.last_item(), "__weight__theme2")

                with dpg.theme(tag="__data__theme2"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(6/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(6/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(6/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                dpg.add_button(label="Data", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.bind_item_theme(dpg.last_item(), "__data__theme2")

                with dpg.theme(tag="__save__theme2"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(6/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(6/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(6/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                dpg.add_button(label="Save", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.bind_item_theme(dpg.last_item(), "__save__theme2")

            with dpg.tree_node(label="Method", default_open=True):
                options = ["SORT", "StrongSORT", "OCSORT", "BoTSORT", "SDTrack"]
                dpg.add_radio_button(label="Choose a tracking method", items=options, default_value="SDTrack", horizontal=True)

            with dpg.tree_node(label="config", default_open=True):
                dpg.add_checkbox(label="CMC")
                _help("User Camera Movement Compensation(CMC) or not")
                with dpg.group(horizontal=True):
                    dpg.add_text("Cascade Levels: ")
                    widget = dpg.add_text("3")
                    dpg.add_button(arrow=True, direction=dpg.mvDir_Left, user_data=widget, callback=lambda s, a, u: dpg.set_value(u, int(dpg.get_value(u))-1))
                    dpg.add_button(arrow=True, direction=dpg.mvDir_Right, user_data=widget, callback=lambda s, a, u: dpg.set_value(u, int(dpg.get_value(u))+1))
                dpg.add_input_float(label="Confidence Threshold", format="%.2f", default_value=0.6)
                dpg.add_input_float(label="IoU Threshold", format="%.2f", default_value=0.6)
                
            with dpg.theme(tag="__track__theme"):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(4/7.0, 0.6, 0.6))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(4/7.0, 0.8, 0.8))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(4/7.0, 0.7, 0.7))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3 * 3, 3 * 3)
            
            dpg.add_button(label="Start Tracking", callback=track)
            dpg.bind_item_theme(dpg.last_item(), "__track__theme")

        with dpg.collapsing_header(label="Detect", default_open=True):

            with dpg.group(horizontal=True):
                with dpg.theme(tag="__weight__theme3"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(5/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(5/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(5/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                dpg.add_button(label="Weight", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.bind_item_theme(dpg.last_item(), "__weight__theme3")

                with dpg.theme(tag="__data__theme3"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(5/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(5/7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(5/7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                dpg.add_button(label="Data", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.bind_item_theme(dpg.last_item(), "__data__theme3")

                with dpg.theme(tag="__save__theme3"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(5/7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(5/7.0, 0.9, 0.9))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(5/7.0, 0.8, 0.8))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
                dpg.add_button(label="Save", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.bind_item_theme(dpg.last_item(), "__save__theme3")
            
            
           
            dpg.add_input_float(label="IoU Threshold", format="%.2f", default_value=0.4)
            dpg.add_checkbox(label="Show Confidence",default_value=True)
            dpg.add_checkbox(label="Show Labels", default_value=True)
            
            with dpg.theme(tag="__detect__theme"):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(4/7.0, 0.5, 0.5))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(4/7.0, 0.9, 0.9))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(4/7.0, 0.8, 0.8))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3 * 3, 3 * 3)
            
            dpg.add_button(label="Start Detecting", callback=detect)
            dpg.bind_item_theme(dpg.last_item(), "__detect__theme")
            

        with dpg.collapsing_header(label="Mode", default_open=True):
            dpg.add_checkbox(label="Counting", default_value=True)
            _help("count and print the detected objects in order")
            dpg.add_checkbox(label="Postion")
            _help("print the pixel position of objects of certain classes")
            dpg.add_color_edit((255, 0, 0, 255), label="choose color")
            _help(
                    "Click on the colored square to open a color picker.\n"
                    "Click and hold to use drag and drop.\n"
                    "Right-click on the colored square to show options.\n"
                    "CTRL+click on individual component to input value.\n")
     

if __name__ == '__main__':
    
    dpg.create_context()
    dpg.create_viewport(title='Detect and Track', width=2500, height=1200)

    show_main()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()