"""文件处理和保存功能."""

import os
from tkinter import filedialog, messagebox

import cv2


class FileHandler:
    """文件处理类."""

    @staticmethod
    def select_file():
        """选择文件."""
        file_path = filedialog.askopenfilename(
            title="选择图片或视频文件",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("所有文件", "*.*"),
            ],
        )
        return file_path

    @staticmethod
    def is_video_file(file_path):
        """判断是否为视频文件."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

    @staticmethod
    def load_image_preview(file_path):
        """加载图片预览."""
        try:
            img = cv2.imread(file_path)
            return img
        except:
            return None

    @staticmethod
    def save_detection_results(
        detection_results, detection_file_type, selected_file_path, add_info_callback, show_message_callback
    ):
        """保存检测结果."""
        # 检查是否有检测结果
        if detection_results is None:
            messagebox.showinfo("提示", "没有检测结果可保存")
            return False

        # 对于视频，检查列表是否为空
        if detection_file_type == "video" and len(detection_results) == 0:
            messagebox.showinfo("提示", "没有检测结果可保存")
            return False

        try:
            if detection_file_type == "image":
                # 图片文件：让用户选择保存位置
                default_filename = f"detected_{os.path.basename(selected_file_path)}"

                save_path = filedialog.asksaveasfilename(
                    title="保存检测结果",
                    defaultextension=".jpg",
                    filetypes=[("JPEG文件", "*.jpg"), ("PNG文件", "*.png"), ("所有文件", "*.*")],
                    initialfile=default_filename,
                )

                if save_path:
                    cv2.imwrite(save_path, detection_results)
                    add_info_callback(f"已将检测结果保存至: {save_path}")
                    show_message_callback("成功", f"检测结果已保存至:\n{save_path}")
                    return True

            elif detection_file_type == "video":
                # 视频文件：让用户选择保存文件夹
                save_dir = filedialog.askdirectory(title="选择保存文件夹")

                if save_dir:
                    # 获取原始文件名
                    original_filename = os.path.basename(selected_file_path)
                    name_without_ext = os.path.splitext(original_filename)[0]
                    output_filename = f"detected_{name_without_ext}.mp4"
                    output_path = os.path.join(save_dir, output_filename)

                    # 获取视频帧的尺寸
                    if len(detection_results) > 0:
                        height, width = detection_results[0].shape[:2]

                        # 创建视频写入器
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

                        # 写入所有帧
                        for frame in detection_results:
                            out.write(frame)

                        out.release()
                        add_info_callback(f"已将检测结果保存至: {output_path}")
                        show_message_callback("成功", f"检测结果已保存至:\n{output_path}")
                        return True
                    else:
                        messagebox.showerror("错误", "没有视频帧可保存")
                        return False

        except Exception as e:
            add_info_callback(f"保存检测结果时发生错误: {e!s}")
            messagebox.showerror("错误", f"保存检测结果时发生错误: {e!s}")
            return False

        return False
