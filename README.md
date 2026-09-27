# clinmock

臨床データをローカルで要約統計（profile）に変換し、その profile から schema・周辺分布・おおまかな変数間依存が近い dummy dataset を生成するためのライブラリです。実データはローカルに残し、dummy data だけを coding agent に渡す運用を想定しています。

**状態: V0.1 開発中（M0–M1 完了）。** 現在使えるのは型推定と Profile JSON の保存・読み込みだけです。`profile_dataframe` / `generate` / `compare` は後続のマイルストーンで追加します。

## 開発環境

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
```

## 型推定

```python
import clinmock
from clinmock.datasets import make_clinical_like_df

df = make_clinical_like_df(n=10000, seed=0)   # 乱数だけで作った臨床風データ
clinmock.infer_types(df)                       # {"age": "numeric", "ecog": "ordinal", ...}
```

整数コード列（1/2/3 など）は `types=` で型を明示することを推奨します。指定がない場合、値が {0,1} なら binary、distinct 値が 10 個以下なら ordinal と推定し、警告を出します。

```python
types = {"stage": {"type": "ordinal", "levels": ["I", "II", "III", "IV"]},
         "ecog": "ordinal"}
resolution = clinmock.resolve_types(df, types, pseudonymize=["hospital", "doctor"])
```

## level 名の仮名化（opt-in）

文字列カテゴリの level 名は、既定ではそのまま profile に保存します。仮名（`L01`, `L02`, …）に置き換えたい列は、次のどちらかで明示的に指定してください。両方で指定した場合は列ごとの指定が優先されます。

- `pseudonymize=["hospital", "doctor"]`
- 列ごとの指定: `types={"hospital": {"type": "nominal", "pseudonymize": True}}`

整数コードや bool/0-1 の列は仮名化しません（指定しても警告を出して無視します）。

仮名化するかどうかを clinmock が自動で決めることはなく、判断はユーザーに委ねます。ただし、仮名化していない文字列カテゴリ列が次のどちらかに当てはまる場合は `PrivacyWarning`（code `consider_pseudonymize`）を出すので、内容を確認してください。

- 列名が識別子・人・施設を示す典型パターンに一致する（id, name, hospital, doctor, facility, site, center, institution, 病院, 医師 など）
- 値のほぼすべて（90% 以上）が行ごとに異なる

## Profile JSON

スキーマ version は `0.1.0` です。仕様は `docs/`（プロジェクトの計画書 §3）にあります。仮名と実名の対応表（`LevelMap`）は profile JSON には含めません。`profile.save_level_map("level_map.local.json")` で別ファイルに書き出し、ローカルだけで使ってください。`*.local.json` は `.gitignore` 済みです。
