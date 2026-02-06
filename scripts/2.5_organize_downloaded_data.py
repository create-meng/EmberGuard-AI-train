"""
整理下载的数据集
将下载的视频文件整理到统一的目录结构中
"""
import os
import shutil
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_section(text):
    """打印章节"""
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}\n")

def analyze_downloaded_data():
    """分析已下载的数据"""
    print_header("📊 分析已下载的数据集")
    
    download_dir = Path("datasets/download")
    
    datasets_found = {
        "mivia_fire": {
            "path": download_dir / "mivia_fire/mivia_fire",
            "count": 0,
            "type": "fire",
            "label": 2
        },
        "mivia_smoke": {
            "path": download_dir / "mivia_smoke/mivia_smoke",
            "count": 0,
            "type": "smoke/normal",
            "label": "mixed"
        },
        "archive": {
            "path": download_dir / "archive/data/video_data",
            "count": 0,
            "type": "mixed",
            "label": "mixed"
        },
        "fire_smoke_github": {
            "path": download_dir / "Fire-Smoke-Dataset-master/Assets",
            "count": 0,
            "type": "images",
            "label": "N/A"
        }
    }
    
    # 统计文件数量
    for name, info in datasets_found.items():
        if info["path"].exists():
            if name == "mivia_fire":
                info["count"] = len(list(info["path"].glob("*.avi")))
            elif name == "mivia_smoke":
                # 统计所有子目录中的avi文件
                info["count"] = len(list(info["path"].rglob("*.avi")))
            elif name == "archive":
                # 统计train和test视频
                train_videos = list((info["path"] / "train_videos").glob("*.mp4"))
                test_videos = list((info["path"] / "test_videos").glob("*.mp4"))
                info["count"] = len(train_videos) + len(test_videos)
            elif name == "fire_smoke_github":
                info["count"] = len(list(info["path"].glob("*.jpg")))
    
    # 打印统计结果
    print("发现的数据集:")
    print()
    total_videos = 0
    for name, info in datasets_found.items():
        if info["count"] > 0:
            print(f"✅ {name}")
            print(f"   路径: {info['path']}")
            print(f"   文件数: {info['count']}")
            print(f"   类型: {info['type']}")
            print()
            if name != "fire_smoke_github":  # 不计算图片
                total_videos += info["count"]
    
    print(f"总视频数: {total_videos}")
    
    return datasets_found

def create_organized_structure():
    """创建整理后的目录结构"""
    print_section("创建目录结构")
    
    base_dir = Path("datasets/fire_videos_organized")
    dirs = {
        "fire": base_dir / "fire",
        "smoke": base_dir / "smoke",
        "normal": base_dir / "normal",
        "mixed": base_dir / "mixed"  # 需要手动分类的
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {path}")
    
    return base_dir

def copy_mivia_fire(base_dir):
    """复制MIVIA火灾视频"""
    print_section("整理 MIVIA Fire Dataset")
    
    source_dir = Path("datasets/download/mivia_fire/mivia_fire")
    target_dir = base_dir / "fire"
    
    if not source_dir.exists():
        print("❌ 未找到MIVIA火灾数据集")
        return 0
    
    count = 0
    for video_file in source_dir.glob("*.avi"):
        target_file = target_dir / f"mivia_fire_{video_file.name}"
        shutil.copy2(video_file, target_file)
        count += 1
        if count % 10 == 0:
            print(f"  已复制 {count} 个文件...")
    
    print(f"✅ 复制了 {count} 个火灾视频到 {target_dir}")
    return count

def copy_mivia_smoke(base_dir):
    """复制MIVIA烟雾视频"""
    print_section("整理 MIVIA Smoke Dataset")
    
    source_dir = Path("datasets/download/mivia_smoke/mivia_smoke")
    
    if not source_dir.exists():
        print("❌ 未找到MIVIA烟雾数据集")
        return 0, 0
    
    # 分类规则
    smoke_dirs = ["SmokeAVI", "Smoke + Red reflection"]
    normal_dirs = ["Clouds", "Mountains", "Sun", "Red reflections"]
    
    smoke_count = 0
    normal_count = 0
    
    # 复制烟雾视频
    for dir_name in smoke_dirs:
        dir_path = source_dir / dir_name
        if dir_path.exists():
            for video_file in dir_path.glob("*.avi"):
                target_file = base_dir / "smoke" / f"mivia_smoke_{video_file.name}"
                shutil.copy2(video_file, target_file)
                smoke_count += 1
    
    print(f"✅ 复制了 {smoke_count} 个烟雾视频")
    
    # 复制正常场景视频
    for dir_name in normal_dirs:
        dir_path = source_dir / dir_name
        if dir_path.exists():
            for video_file in dir_path.glob("*.avi"):
                target_file = base_dir / "normal" / f"mivia_normal_{video_file.name}"
                shutil.copy2(video_file, target_file)
                normal_count += 1
    
    print(f"✅ 复制了 {normal_count} 个正常场景视频")
    
    return smoke_count, normal_count

def copy_archive_videos(base_dir):
    """复制archive数据集视频"""
    print_section("整理 Archive Dataset")
    
    source_dir = Path("datasets/download/archive/data/video_data")
    
    if not source_dir.exists():
        print("❌ 未找到Archive数据集")
        return 0, 0, 0
    
    fire_count = 0
    smoke_count = 0
    normal_count = 0
    
    # 训练视频
    train_dir = source_dir / "train_videos"
    if train_dir.exists():
        for video_file in train_dir.glob("*.mp4"):
            filename = video_file.name.lower()
            
            if "fire" in filename and "smoke" not in filename:
                # 纯火灾视频
                target_file = base_dir / "fire" / f"archive_{video_file.name}"
                shutil.copy2(video_file, target_file)
                fire_count += 1
            elif "smoke" in filename:
                # 烟雾视频
                target_file = base_dir / "smoke" / f"archive_{video_file.name}"
                shutil.copy2(video_file, target_file)
                smoke_count += 1
            elif "nofire" in filename or "normal" in filename:
                # 正常场景
                target_file = base_dir / "normal" / f"archive_{video_file.name}"
                shutil.copy2(video_file, target_file)
                normal_count += 1
            else:
                # 混合或不确定的，放到mixed目录
                target_file = base_dir / "mixed" / f"archive_{video_file.name}"
                shutil.copy2(video_file, target_file)
    
    # 测试视频（放到mixed，需要手动分类）
    test_dir = source_dir / "test_videos"
    if test_dir.exists():
        for video_file in test_dir.glob("*.mp4"):
            target_file = base_dir / "mixed" / f"archive_test_{video_file.name}"
            shutil.copy2(video_file, target_file)
    
    print(f"✅ 火灾视频: {fire_count}")
    print(f"✅ 烟雾视频: {smoke_count}")
    print(f"✅ 正常视频: {normal_count}")
    
    return fire_count, smoke_count, normal_count

def copy_bowfire_videos(base_dir):
    """复制BoWFire数据集视频"""
    print_section("整理 BoWFire Dataset (836749)")
    
    bowfire_dir = Path("datasets/download/836749")
    
    if not bowfire_dir.exists():
        print("❌ 未找到BoWFire数据集")
        return 0, 0, 0
    
    fire_count = 0
    smoke_count = 0
    normal_count = 0
    
    # 火灾视频
    fire_pos_dir = bowfire_dir / "fire_videos.1406/pos"
    if fire_pos_dir.exists():
        for video_file in fire_pos_dir.glob("*.avi"):
            target_file = base_dir / "fire" / f"bowfire_{video_file.name}"
            shutil.copy2(video_file, target_file)
            fire_count += 1
    
    # 非火灾视频（正常场景）
    fire_neg_dir = bowfire_dir / "fire_videos.1406/neg"
    if fire_neg_dir.exists():
        for video_file in fire_neg_dir.glob("*.avi"):
            target_file = base_dir / "normal" / f"bowfire_nofire_{video_file.name}"
            shutil.copy2(video_file, target_file)
            normal_count += 1
    
    # 烟雾视频
    smoke_pos_dir = bowfire_dir / "smoke_videos.1407/pos"
    if smoke_pos_dir.exists():
        for video_file in smoke_pos_dir.glob("*.avi"):
            target_file = base_dir / "smoke" / f"bowfire_{video_file.name}"
            shutil.copy2(video_file, target_file)
            smoke_count += 1
    
    # 非烟雾视频（正常场景）
    smoke_neg_dir = bowfire_dir / "smoke_videos.1407/neg"
    if smoke_neg_dir.exists():
        for video_file in smoke_neg_dir.glob("*.avi"):
            target_file = base_dir / "normal" / f"bowfire_nosmoke_{video_file.name}"
            shutil.copy2(video_file, target_file)
            normal_count += 1
    
    print(f"✅ 火灾视频: {fire_count}")
    print(f"✅ 烟雾视频: {smoke_count}")
    print(f"✅ 正常视频: {normal_count}")
    
    return fire_count, smoke_count, normal_count

def create_annotations_csv(base_dir, stats):
    """创建标注CSV文件"""
    print_section("创建标注文件")
    
    csv_file = base_dir / "annotations.csv"
    
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("video_path,label,label_name,source,notes\n")
        
        # 火灾视频
        fire_dir = base_dir / "fire"
        for video_file in sorted(fire_dir.glob("*")):
            rel_path = video_file.relative_to(base_dir)
            source = "mivia" if "mivia" in video_file.name else "archive"
            f.write(f"{rel_path},2,fire,{source},\n")
        
        # 烟雾视频
        smoke_dir = base_dir / "smoke"
        for video_file in sorted(smoke_dir.glob("*")):
            rel_path = video_file.relative_to(base_dir)
            source = "mivia" if "mivia" in video_file.name else "archive"
            f.write(f"{rel_path},1,smoke,{source},\n")
        
        # 正常视频
        normal_dir = base_dir / "normal"
        for video_file in sorted(normal_dir.glob("*")):
            rel_path = video_file.relative_to(base_dir)
            source = "mivia" if "mivia" in video_file.name else "archive"
            f.write(f"{rel_path},0,normal,{source},\n")
        
        # 混合视频（需要手动标注）
        mixed_dir = base_dir / "mixed"
        for video_file in sorted(mixed_dir.glob("*")):
            rel_path = video_file.relative_to(base_dir)
            f.write(f"{rel_path},-1,unknown,archive,需要手动标注\n")
    
    print(f"✅ 创建标注文件: {csv_file}")
    print(f"\n标注统计:")
    print(f"  火灾视频 (标签2): {stats['fire']}")
    print(f"  烟雾视频 (标签1): {stats['smoke']}")
    print(f"  正常视频 (标签0): {stats['normal']}")
    print(f"  待标注 (标签-1): {stats['mixed']}")

def main():
    """主函数"""
    print_header("🔥 EmberGuard AI - 数据集整理工具")
    
    # 分析数据
    datasets = analyze_downloaded_data()
    
    input("\n按Enter键开始整理数据...")
    
    # 创建目录结构
    base_dir = create_organized_structure()
    
    # 统计
    stats = {
        "fire": 0,
        "smoke": 0,
        "normal": 0,
        "mixed": 0
    }
    
    # 复制MIVIA火灾视频
    stats["fire"] += copy_mivia_fire(base_dir)
    
    # 复制MIVIA烟雾视频
    smoke_count, normal_count = copy_mivia_smoke(base_dir)
    stats["smoke"] += smoke_count
    stats["normal"] += normal_count
    
    # 复制Archive视频
    fire_count, smoke_count, normal_count = copy_archive_videos(base_dir)
    stats["fire"] += fire_count
    stats["smoke"] += smoke_count
    stats["normal"] += normal_count
    
    # 复制BoWFire视频
    fire_count, smoke_count, normal_count = copy_bowfire_videos(base_dir)
    stats["fire"] += fire_count
    stats["smoke"] += smoke_count
    stats["normal"] += normal_count
    
    # 统计mixed目录
    mixed_dir = base_dir / "mixed"
    stats["mixed"] = len(list(mixed_dir.glob("*")))
    
    # 创建标注文件
    create_annotations_csv(base_dir, stats)
    
    print_header("✅ 数据整理完成！")
    
    print(f"\n整理后的目录: {base_dir}")
    print(f"\n总计:")
    print(f"  火灾视频: {stats['fire']}")
    print(f"  烟雾视频: {stats['smoke']}")
    print(f"  正常视频: {stats['normal']}")
    print(f"  待标注: {stats['mixed']}")
    print(f"  总计: {sum(stats.values())}")
    
    print(f"\n下一步:")
    print(f"  1. 检查 {base_dir}/mixed/ 目录中的视频")
    print(f"  2. 手动将它们移动到正确的类别目录")
    print(f"  3. 更新 annotations.csv 文件")
    print(f"  4. 运行: python scripts/3_prepare_lstm_data.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
