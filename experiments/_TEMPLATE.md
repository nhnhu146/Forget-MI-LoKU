# Exp {{ID}} — {{NAME}}

| Field | Value |
|---|---|
| **Date** | {{DATE}} |
| **Git commit** | `{{COMMIT}}` |
| **Branch** | `{{BRANCH}}` |
| **Status** | 🔄 Running |
| **Duration** | <!-- AUTO:duration -->(auto)<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

{{HYPOTHESIS}}

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
_(script sẽ tự điền diff config vs exp trước)_
<!-- /AUTO -->

### Code changes (điền thủ công nếu có sửa .py)
- _(file:line) Mô tả_

---

## 3. Full configuration snapshot

<details>
<summary>config.yaml</summary>

```yaml
{{CONFIG}}
```

</details>

---

## 4. Execution

```bash
python training/forgetmi_loku.py --config config.yaml --fresh --exp {{NAME}}
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
_(auto)_
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
_(auto)_
<!-- /AUTO -->

---

## 6. Observations (✍️ điền thủ công)

_(Quan sát gì bất thường? Khớp/khác predict?)_

-

---

## 7. Conclusion (✍️ điền thủ công)

- **Hypothesis verdict**: ✅ Confirmed / ❌ Rejected / 🤷 Inconclusive
- **Keep changes**: Y / N
- **Why**:

---

## 8. Next steps (✍️ điền thủ công)

-
