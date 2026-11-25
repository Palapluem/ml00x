# CPE 342 Term Project — Technical Report (`report/`)

แพ็กเกจรายงานทางวิชาการฉบับสมบูรณ์สำหรับโครงงาน **CPE342 Term Project: Karena Player Intelligence System** ของกลุ่ม **ML004** (ภาคเรียนที่ 1 ปีการศึกษา 2569) จัดทำด้วย XeLaTeX (ฟอนต์ `TH Sarabun New` `Scale=1.46` + `Consolas`) ปราศจากหัวกระดาษ ตามรูปแบบเอกสารทางวิชาการภาควิชาวิศวกรรมคอมพิวเตอร์ มจธ.

## โครงสร้างไฟล์

```text
report/
├── .latexmkrc                       # ตั้งค่า XeLaTeX ให้ส่ง aux ไปยัง TMPDIR และ PDF ไปยัง output/pdf/
├── README.md                        # เอกสารแนะนำโครงสร้างและวิธีคอมไพล์รายงาน
├── CPE342_TermProject_Report.tex    # ต้นฉบับรายงาน XeLaTeX ฉบับสมบูรณ์ (สารบัญ, บทที่ 1–8, บรรณานุกรม)
├── frontpage/
│   ├── frontpage.tex                # หน้าปกรายงาน (โลโก้ KMUTT, CPE342 Term Project, สมาชิกกลุ่ม ML004)
│   └── assets/
│       └── KMUTT_Logo.png
├── assets/                          # กราฟวินิจฉัยความละเอียด 300 dpi (19 ภาพจาก Tasks 1–5 และ Leaderboard)
└── output/
    └── pdf/
        └── CPE342_TermProject_Report.pdf
```

## คำสั่งคอมไพล์ PDF

```sh
cd report
latexmk CPE342_TermProject_Report.tex
```

ไฟล์ `.latexmkrc` จะรัน `xelatex` และแยกไฟล์ชั่วคราว (`.aux`, `.log`, `.out`, `.toc`, `.xdv`) ไปยัง `$TMPDIR/cpe342-report-aux` โดยอัตโนมัติ พร้อมบันทึกผลลัพธ์ลงที่ `output/pdf/CPE342_TermProject_Report.pdf`

## การเชื่อมโยงหลักฐานในคลังซอร์สโค้ดหลัก
- **ผลการรวมคำตอบและคะแนน Leaderboard:** [`submit.md`](../submit.md) และสคริปต์ตรวจรวมผล [`submission/merge.py`](../submission/merge.py)
- **โมดูลซอร์สโค้ดหลักทั้ง 5 งาน:** [`task1/`](../task1/), [`task2/`](../task2/), [`task3/`](../task3/), [`task4/`](../task4/) และ [`task5/`](../task5/)
- **เอกสารข้อกำหนดโครงงาน:** [`docs/project-instruction.md`](../docs/project-instruction.md)
