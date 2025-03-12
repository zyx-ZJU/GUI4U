import dearpygui.dearpygui as dpg

# 你想要执行的函数
def sample_function():
    return "Success"

# 定义回调函数，当按钮点击时输出结果
def button_callback(sender, app_data):
    result = sample_function()
    dpg.set_value("result_text", result)

# 创建窗口
dpg.create_context()

# 创建一个窗口
with dpg.handler_registry():
    # 按钮，点击时执行回调
    with dpg.window(label="Function Output Window"):
        dpg.add_button(label="Run", callback=button_callback)
        # 用于显示函数执行结果的文本框
        dpg.add_text(default_value="Results: ", tag="result_text")

# 创建一个主循环来显示窗口
dpg.create_viewport(title='Output Window', width=400, height=200)
dpg.setup_dearpygui()
dpg.show_viewport()

# 运行窗口
dpg.start_dearpygui()

# 清理资源
dpg.destroy_context()
