# CPE 342 Term Project — Technical Report (`report/`)

แพ็กเกจรายงานทางวิชาการฉบับสมบูรณ์สำหรับโครงงาน **CPE342 Term Project: Karena Player Intelligence System** ของกลุ่ม **ML004** (ภาคเรียนที่ 1 ปีการศึกษา 2569) จัดทำด้วย XeLaTeX (ฟอนต์ `TH Sarabun New` `Scale=1.46` + `Consolas`) ปราศจากหัวกระดาษ ตามรูปแบบเอกสารทางวิชาการภาควิชาวิศวกรรมคอมพิวเตอร์ มจธ.

## โครงสร้างไฟล์

```text
report/
├── .latexmkrc                       # การตั้งค่า XeLaTeX คอมไพล์ลง temp directory ไม่ทิ้งไฟล์ขยะใน repo
├── README.md                        # เอกสารแนะนำโครงสร้างและวิธีคอมไพล์รายงาน
├── frontpage/
│   ├── frontpage.tex                # หน้าปกรายงาน (โลโก้ KMUTT, CPE342 Term Project, สมาชิกกลุ่ม ML004)
│   ├── frontpage.example.tex        # ตัวอย่างต้นแบบหน้าปกเดิม
│   └── assets/
│       ├── KMUTT_Logo.png
│       └── KMUTT_Logo.svg
├── assets/                          # กราฟวินิจฉัยความละเอียด 300 dpi (19 ภาพจาก Tasks 1–5 และ Leaderboard)
├── report.tex                       # ต้นฉบับรายงาน XeLaTeX ฉบับสมบูรณ์ (สารบัญ, บทที่ 1–8, บรรณานุกรม)
└── output/
    └── pdf/
        └── CPE342_TermProject_Report.pdf
```

## คำสั่งคอมไพล์ PDF

```sh
cd report
build_dir="$(mktemp -d -t ml00x-report)"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$build_dir" -jobname=CPE342_TermProject_Report report.tex
cp "$build_dir/CPE342_TermProject_Report.pdf" output/pdf/CPE342_TermProject_Report.pdf
rm -rf "$build_dir"
```

## การเชื่อมโยงหลักฐานในคลังซอร์สโค้ดหลัก
- **ผลการรวมคำตอบและคะแนน Leaderboard:** [`submit.md`](../submit.md) และ [`submission/merge.py`](../submission/merge.py)
- **โมดูลซอร์สโค้ดหลักทั้ง 5 งาน:** [`task1/`](../task1/), [`task2/`](../task2/), [`task3/`](../task3/), [`task4/`](../task4/) และ [`task5/`](../task5/)
- **เอกสารข้อกำหนดโครงงาน:** [`docs/project-instruction.md`](../docs/project-instruction.md)
