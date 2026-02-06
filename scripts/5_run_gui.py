"""
运行YOLO火灾检测GUI应用程序
"""
if __name__ == "__main__":
    import sys
    import os
    
    # 设置使用项目本地的Ultralytics配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['ULTRALYTICS_CONFIG_DIR'] = os.path.join(project_root, 'configs')
    
    # 确保项目根目录在Python路径中
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 导入并运行
    from UI.main import main
    main()
