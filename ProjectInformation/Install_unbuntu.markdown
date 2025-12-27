好，下面是**【最终版 · 可直接复制给队友】的 Ubuntu 22.04 LTS 安装教程**。
这一版 **已按你“1TB 三星空硬盘 + 深度学习/5070”场景重写**，
**分区方案明确、空间合理、不玩花活**。

---

# Ubuntu 22.04 LTS 安装教程（最终版｜1TB 空硬盘推荐）

> **适用对象**
>
> * Windows 电脑
> * 额外一块 **1TB 空 SSD（如三星）**
> * 已有 Ubuntu 22.04 LTS ISO
> * 用于 NVIDIA GPU / 深度学习 / 比赛项目
> * **不影响原 Windows**

---

## 一、安装前准备

### 需要的东西

* Windows 电脑一台
* **1TB 空硬盘（不要有数据）**
* Ubuntu 22.04 LTS ISO 文件
* U 盘 ≥ 8GB
* 可联网（非必须）

---

### Windows 下下载工具

**Rufus（制作启动盘）**

* 官网：[https://rufus.ie](https://rufus.ie)
* 下载 **Rufus Portable（免安装版）**

---

## 二、制作 Ubuntu 启动 U 盘

1. 插入 U 盘（⚠ 会清空）
2. 打开 Rufus
3. 设置如下：

| 项目   | 设置                   |
| ---- | -------------------- |
| 设备   | 你的 U 盘               |
| 启动类型 | Ubuntu 22.04 LTS ISO |
| 分区类型 | GPT                  |
| 目标系统 | UEFI（非 CSM）          |
| 文件系统 | FAT32                |
| 其他   | 全部默认                 |

4. 点击 **START**
5. 弹窗选择 **ISO 模式（推荐）**
6. 等待完成

---

## 三、从 U 盘启动（BIOS 设置）

1. 重启电脑
2. 开机时连续按以下任一键：

`F2 / F12 / DEL`

3. BIOS 中确认：

   * **Boot Mode = UEFI**
   * **Secure Boot = Disabled**
4. 选择启动项：

   ```
   UEFI: USB Storage
   ```

---

## 四、进入安装界面

出现 Ubuntu 菜单后：

* 选择 **Try or Install Ubuntu**
* 等待进入安装器

---

## 五、基础安装选项

### 1️⃣ 语言

* **English（推荐）**

### 2️⃣ 键盘

* English (US)

### 3️⃣ 网络

* 有网就连
* 没网可跳过

### 4️⃣ 安装类型

* **Normal Installation**
* 勾选：

  * ☑ Install third-party software

---

## 六、磁盘分区（关键步骤）

### ⚠ 必须选择：

👉 **Something else（手动分区）**

❌ 不要选：

* Erase disk and install Ubuntu

---

### 确认硬盘

找到 **1TB 空硬盘**（示例）：

```
/dev/nvme1n1   ≈ 1TB   （三星 SSD）
```

⚠ 不要选 Windows 硬盘

---

## 七、1TB 空硬盘推荐分区方案（照着点）

### ① EFI 分区（启动用）

* Size：`512 MB`
* Type：EFI System Partition
* Mount point：`/boot/efi`

---

### ② 根分区 `/`（核心）

* Size：**除 EFI + swap 外的全部空间**

  * 约 `980 GB`
* Type：Ext4
* Mount point：`/`

> 系统 / CUDA / PyTorch / TensorRT / 模型 / 数据
> **全部放这里**

---

### ③ swap 分区（推荐）

* Size：

  * **16 GB（最低）**
  * **32 GB（内存 ≤ 32GB 推荐）**
* Type：swap

---

### 启动器安装位置（非常重要）

选择：

```
/dev/nvme1n1   （整块 1TB 硬盘）
```

❌ 不选分区
❌ 不选 Windows 硬盘

---

## 八、开始安装

1. 点击 **Install Now**
2. 确认分区
3. 选择时区
4. 设置用户名 / 密码
5. 等待 10–20 分钟

---

## 九、安装完成

1. 点击 **Restart Now**
2. 提示拔 U 盘 → 拔掉 → 回车
3. 成功进入 Ubuntu 桌面

---

## 十、首次进入系统必做

打开 Terminal，执行：

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

---

## 十一、确认分区是否正确

重启后执行：

```bash
df -h
```

看到类似：

```
/dev/nvme1n1p2   ~900G   ...   /
```

说明分区正确。

---

## 十二、完成标准

* Ubuntu 可正常进入
* Windows 不受影响
* 根分区空间充足（≈1TB）

👉 **系统安装完成**

---

### 后续步骤（不要自己乱搞）

下一步必须做的是：

* NVIDIA 显卡驱动（适配 RTX 5070）
* CUDA + PyTorch

如果需要，可以继续提供 **“5070 最稳显卡驱动 + CUDA 安装教程”**。
