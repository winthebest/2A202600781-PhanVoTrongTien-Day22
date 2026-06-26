# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _<Phan Võ Trọng Tiển>_
**Cohort:** _<2A202600781>_
**Tier đã chạy:** T4
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Kaggle/Colab **Tesla T4 16GB** (15.6 GB usable) |
| CUDA / driver | CUDA 12.x (T4 / Turing, fp16 — bf16 not supported) |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit, 3.12B params) |
| SFT dataset slice | Vietnamese-Alpaca · 1000 samples · 1 epoch (125 steps) |
| Preference dataset slice | UltraFeedback (binarized, English baseline) · 2000 pairs · 1 epoch (250 steps) |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free tier) |

LoRA: r=16, alpha=32 → trainable 29,933,568 / 3,115,872,256 params (0.96%).

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _<điền: phút từ output train>_ |
| VRAM peak | _<~10 GB>_ | _<~13–14 GB (ref model doubles VRAM)>_ |
| Final loss | 1.3425 (SFT-mini) | 1.6205 (DPO) |
| Reward gap (chosen − rejected, end) | n/a | **−0.520** (chosen +1.179, rejected +1.699) |
| Win-rate vs SFT (8 prompts, gpt-4o-mini judge) | 3/8 | **5/8** |

**Tulu 3 reference numbers** (deck §7.2b, context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO, Llama-3-8B-Instruct, 70B-class). Không kỳ vọng tái lập ở 3B.

---

## 3. Reward curves analysis (≥ 100 words)

> Xem `submission/screenshots/03-dpo-reward-curves.png`.

Cuối training: **chosen reward = +1.179**, **rejected reward = +1.699**, nên **gap = −0.520 (âm)**. Điểm mấu chốt khi đọc cả hai đường: đây **không phải likelihood displacement** theo deck §3.4 — likelihood displacement là khi *chosen reward GIẢM (đi âm)* trong khi gap vẫn nới ra. Ở đây chosen reward **dương và tăng** so với khởi điểm ~0; vấn đề là rejected reward tăng *nhanh hơn*. Nói cách khác, mô hình đẩy log-prob của **cả** chosen lẫn rejected lên, nhưng không tách được hai phân phối — margin không mở ra, thậm chí đảo dấu. Nguyên nhân khả dĩ: (1) tập preference là **UltraFeedback tiếng Anh** trong khi base đã SFT trên dữ liệu **tiếng Việt**, nên tín hiệu chosed/rejected không khớp domain; (2) β=0.1 có thể quá chặt/lỏng cho cặp dữ liệu này; (3) chỉ 1 epoch / 250 step — chưa đủ để margin tách. Điều thú vị: **dù gap training âm, win-rate downstream vẫn 5/8 nghiêng về DPO** (đặc biệt safety 3/1), cho thấy cải thiện hành vi xuất hiện ở generation ngay cả khi reward gap implicit chưa hội tụ — một disconnect đáng lưu ý giữa metric training và đánh giá thực tế.

---

## 4. Qualitative comparison (≥ 8 examples)

> Xem `submission/screenshots/04-side-by-side-table.png`. Bảng đầy đủ 8 prompt × 2 model nằm trong NB4.

**Win/loss/tie summary:** SFT+DPO thắng **5/8**, SFT-only 3/8, tie 0/8.
- Helpfulness: SFT 2/4 — DPO 2/4 (hòa)
- Safety: SFT 1/4 — **DPO 3/4** (DPO cải thiện rõ ở mảng an toàn)

**Judge used:** gpt-4o-mini (API judge, có OPENAI_API_KEY).

_<Tùy chọn: chép 8 dòng prompt/category/winner từ bảng NB4 vào đây cho người chấm tiện đối chiếu.>_

---

## 5. β trade-off

Mình **không** chạy β-sweep (rigor add-on). Hypothesis dựa trên kết quả gap âm ở β=0.1:
- **β nhỏ hơn (0.05):** ràng buộc KL lỏng hơn → policy được phép lệch xa reference hơn → kỳ vọng margin chosen−rejected mở rộng hơn, nhưng rủi ro reward hacking / output dài bất thường.
- **β lớn hơn (0.5):** giữ sát reference → an toàn nhưng gần như không tách được chosen/rejected (gap còn nhỏ hơn). Với hiện tượng gap âm hiện tại, mình dự đoán **giảm β về 0.05** là hướng đáng thử nhất để cứu margin, khớp dự đoán deck §3.3 (β điều tiết đánh đổi giữa bám reference và tối ưu preference).

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> _Bản nháp — hãy đọc lại và viết bằng giọng của chính bạn trước khi nộp._

Quyết định đáng nói nhất của mình là **chọn dùng tập preference UltraFeedback tiếng Anh** thay vì tự build một tập preference tiếng Việt. Phương án thay thế là sinh cặp chosen/rejected tiếng Việt (như BONUS-CHALLENGE gợi ý), nhưng mình chọn UltraFeedback vì nó đã được binarize sẵn, sạch, và là baseline chuẩn của deck — tiết kiệm thời gian trên free T4. Kết quả **vừa xác nhận vừa làm mình bất ngờ**: reward gap training ra **âm (−0.520)**, đúng như rủi ro "lệch domain" mình lo (base SFT tiếng Việt vs preference tiếng Anh), nhưng bất ngờ là **win-rate downstream vẫn 5/8 nghiêng về DPO**, đặc biệt mảng safety 3/1. Điều này dạy mình rằng **reward gap implicit không phải thước đo duy nhất** — phải nhìn cả generation thực tế. Nếu làm lại ngày mai, mình sẽ: (1) build một slice preference **tiếng Việt nhỏ** để khớp domain với SFT, (2) chạy **β-sweep {0.05, 0.1, 0.5}** để xem margin có cứu được không, và (3) tăng lên 2 epoch. Mình kỳ vọng khớp domain sẽ là đòn bẩy lớn nhất để đưa gap về dương.

---

## 7. Benchmark interpretation (≥ 150 words)

Mình **không** chạy NB6 benchmark (add-on bonus, không bắt buộc cho core). Dưới đây là dự đoán định tính dựa trên kết quả NB3/NB4:

- **IFEval:** kỳ vọng DPO **tăng nhẹ** — DPO tinh chỉnh instruction-following, và win-rate helpfulness/safety đã cải thiện.
- **GSM8K / MATH:** rủi ro **alignment tax** (deck §8.1) — DPO trên preference chat có thể làm suy giảm khả năng suy luận toán; với gap âm và chỉ 1 epoch, mình dự đoán gần như **không đổi hoặc giảm rất nhẹ**.
- **MMLU (sampled):** kỳ vọng **gần phẳng** — kiến thức factual nằm ở base weights, LoRA 0.96% params khó gây catastrophic forgetting trong 250 step.
- **AlpacaEval-lite:** kỳ vọng **khớp xu hướng NB4** (DPO ~ 60%+ win-rate), vì cùng dùng judge LLM trên các prompt mở.

Nếu chạy thật, điều mình tò mò nhất là liệu GSM8K có giảm không — đó sẽ là bằng chứng trực tiếp cho alignment tax mà gap-âm này gợi ý.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded)
- [ ] Pair work với: —

---

## Điều ngạc nhiên nhất khi làm lab này

Reward gap training ra **âm** nhưng win-rate downstream vẫn nghiêng về DPO (5/8) — metric training và đánh giá thực tế không phải lúc nào cũng đồng pha.
