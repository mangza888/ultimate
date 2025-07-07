#!/usr/bin/env python3
# cleanup_problematic_files.py - ลบไฟล์ที่มีปัญหา

import os
import shutil

def cleanup_problematic_files():
    """ลบไฟล์ที่มี gpus parameter และสร้างความเสียหาย"""
    
    files_to_remove = [
        './ai/deep_learning_trainer-old.py',
        './ai/pytorch_lightning_fix.py', 
        './ai/deep_learning_models-old2.py',
        './debug_script.py'  # ลบ debug script ด้วย
    ]
    
    print("🧹 CLEANUP: Removing problematic files...")
    print("=" * 50)
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
            else:
                print(f"⚠️  Not found: {file_path}")
        except Exception as e:
            print(f"❌ Error removing {file_path}: {e}")
    
    print("\n🔧 Keeping necessary files...")
    
    # ไฟล์ที่ต้องแก้ไข (ลบ reference ไปยังไฟล์เก่า)
    files_to_check = [
        './ai/reinforcement_learning.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"📝 Check manually: {file_path} (may contain old references)")
    
    print("\n✅ Cleanup completed!")

def backup_current_files():
    """สำรองไฟล์ปัจจุบันก่อนแก้ไข"""
    
    files_to_backup = [
        './ai/deep_learning_models.py',
        './backtesting/advanced_backtesting.py'
    ]
    
    print("💾 BACKUP: Creating backups...")
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = file_path + '.backup'
            try:
                shutil.copy2(file_path, backup_path)
                print(f"✅ Backed up: {file_path} → {backup_path}")
            except Exception as e:
                print(f"❌ Backup failed for {file_path}: {e}")

def check_imports():
    """ตรวจสอบ import ในไฟล์หลัก"""
    
    main_files = [
        './main.py',
        './ai/__init__.py'
    ]
    
    print("\n🔍 CHECKING: Import statements...")
    
    for file_path in main_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            # ตรวจสอบ import ที่เป็นปัญหา
            problematic_imports = [
                'deep_learning_trainer-old',
                'pytorch_lightning_fix',
                'deep_learning_models-old2'
            ]
            
            found_issues = []
            for imp in problematic_imports:
                if imp in content:
                    found_issues.append(imp)
            
            if found_issues:
                print(f"⚠️  {file_path} contains problematic imports: {found_issues}")
            else:
                print(f"✅ {file_path} looks clean")

if __name__ == "__main__":
    print("🚀 CLEANUP PROBLEMATIC FILES")
    print("=" * 50)
    
    # สำรองไฟล์ก่อน
    backup_current_files()
    print()
    
    # ลบไฟล์ที่มีปัญหา
    cleanup_problematic_files()
    print()
    
    # ตรวจสอบ imports
    check_imports()
    
    print("\n" + "=" * 50)
    print("📋 NEXT STEPS:")
    print("1. Replace ai/deep_learning_models.py with clean version")
    print("2. Replace backtesting/advanced_backtesting.py with fixed version") 
    print("3. Run: python main.py")
    print("4. No more 'gpus' parameter errors!")
