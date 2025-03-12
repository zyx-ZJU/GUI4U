import dearpygui.dearpygui as dpg

# 回调函数
def on_selectable_item(sender, app_data):
    # 获取当前选择的项
    selected_item = sender
    print(f"Selected Item: {selected_item}")

# 创建一个窗口


# 启动DearPyGui
dpg.create_context()
dpg.create_viewport(title="DearPyGui - Selectable Example", width=400, height=200)

with dpg.handler_registry():
    with dpg.window(label="Selectable Example"):
        # 创建可选择项
        dpg.add_selectable(label="Option 1", callback=on_selectable_item)
        dpg.add_selectable(label="Option 2", callback=on_selectable_item)
        dpg.add_selectable(label="Option 3", callback=on_selectable_item)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
