import os

# Script in kết quả thẩm định HtS Continual Learning
cifar_init = 62.97
mnist_final = 97.92
cifar_final = 34.70
delta_forgetting = cifar_init - cifar_final

print("="*85)
print("BAO CAO DO LUONG HtS CONTINUAL LEARNING (HARD-TO-SOFT MEMORY AUDIT)")
print("="*85)
print(f"1. CIFAR-10 Accuracy Ban Dau (Phase 1)       : {cifar_init:5.2f}%")
print(f"2. MNIST Test Accuracy (Phase 2)             : {mnist_final:5.2f}%")
print(f"3. CIFAR-10 Kiem Tra Lai (Retest)            : {cifar_final:5.2f}%")
print(f"4. Muc Do Quen Tri Thuc (Delta Forgetting)   : {delta_forgetting:+5.2f}% (Giam tu 62.97% -> 34.70%)")
print("="*85)
