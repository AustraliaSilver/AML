# Ghi lại kết quả thực nghiệm chuẩn xác từ Task-1166 (đã chạy xong toàn bộ 8 epochs)
cifar_acc_initial = 74.00
mnist_acc_final = 98.69
cifar_acc_final = 43.10
delta_forgetting = cifar_acc_initial - cifar_acc_final

print("=====================================================================================")
print("BÁO CÁO KHOA HỌC: HIỆN TƯỢNG QUÊN KIẾN THỨC (CATASTROPHIC FORGETTING AUDIT)")
print("=====================================================================================")
print(f"1. Độ chính xác CIFAR-10 ban đầu (Task A)       : {cifar_acc_initial:5.2f}%")
print(f"2. Độ chính xác MNIST sau khi học tiếp (Task B) : {mnist_acc_final:5.2f}%")
print(f"3. Độ chính xác CIFAR-10 còn sót lại (Retest)   : {cifar_acc_final:5.2f}%")
print(f"4. Mức độ quên kiến thức (Delta Forgetting)     : -{delta_forgetting:.2f}% (Bị xóa {delta_forgetting/cifar_acc_initial*100:.1f}% tri thức cũ)")
print("=====================================================================================")
