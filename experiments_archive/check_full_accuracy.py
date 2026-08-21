import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pure_vision_aml_benchmark import AMLVisionArchitectureSearchModel

def test_full_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)
    
    model = AMLVisionArchitectureSearchModel(dim=192, depth=6, K=16).to(device)
    model.load_state_dict(torch.load("best_vision_aml_arch_search.pth", map_location=device, weights_only=True))
    model.eval()
    
    # Đo đạc 2 chế độ:
    # 1. Soft Dynamic Inference (Inference Thích Ứng Mềm)
    # 2. Hard Discretized Inference (Inference Cứng 100% Khối Được Chọn)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_soft = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, tau=0.3)
            _, pred = logits.max(1)
            total += y.size(0)
            correct_soft += pred.eq(y).sum().item()
            
            c = (pred == y).squeeze()
            for i in range(y.size(0)):
                label = y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    overall_acc = 100.0 * correct_soft / total
    print("="*75)
    print(f"TONG KET DO CHINH XAC (TEST ACCURACY TREN TOAN BO 10,000 ANH CIFAR-10)")
    print("="*75)
    print(f"-> DO CHINH XAC TRUNG BINH TOAN MANG : {overall_acc:.2f}%\n")
    print("CHI TIET DO CHINH XAC TUNG CLASS (10 LOAI DOI TUONG):")
    print("-" * 55)
    for i in range(10):
        acc_cls = 100 * class_correct[i] / class_total[i]
        print(f"  * Class {i} ({classes[i]:<6}) : {acc_cls:5.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    print("="*75)

if __name__ == '__main__':
    test_full_accuracy()
