# KageFrame

*a statistical shadow of your DataFrame*

`kageframe` は、手元の DataFrame（主に臨床データ）を**要約統計だけの profile** に変換し、その profile から、列・型・周辺分布・欠測率・おおまかな変数間の依存が近い **dummy DataFrame** を生成するライブラリです。実データはローカルに残したまま、dummy data だけを使って解析コードを書く・レビューする・coding agent に渡す、という運用を想定しています。

**状態: V0.1 開発中。** `profile_dataframe` / `generate` / `compare` が使えます。numeric・binary・ordinal・nominal・date・datetime・time の全型と欠測（MCAR）に対応しています。V0.1 のリリースまでは API と profile JSON のスキーマ（`0.1.0`）が変わる可能性があります。

## インストール

Python 3.10 以上が必要です。依存は numpy・pandas（2.x 以降）・scipy だけです。PyPI にはまだ公開していないので、GitHub から入れてください。

```bash
pip install "git+https://github.com/mizuy/kageframe.git"
```

開発する場合:

```bash
git clone https://github.com/mizuy/kageframe.git && cd kageframe
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # 全テスト（数秒）
.venv/bin/ruff check .
```

## クイックスタート

```python
import kageframe
from kageframe.datasets import make_clinical_like_df

# 実データの代わりに、乱数だけで作った臨床風データを使う
df = make_clinical_like_df(n=10000, seed=0)

# 1. profile を作る（実データのある環境で実行）
profile = kageframe.profile_dataframe(
    df,
    types={"stage": {"type": "ordinal", "levels": ["I", "II", "III", "IV"]}},
)
profile.save("profile.json")                 # 持ち出すのはこのファイルだけ

# 2. dummy を生成する（profile.json さえあればどこでも実行できる）
profile = kageframe.load_profile("profile.json")
dummy = kageframe.generate(profile, n=10000, seed=42)

# 3. dummy が profile にどれだけ近いかを確かめる
report = kageframe.compare(profile, dummy)
print(report.check())                        # "all checks passed"
report.summary()                             # 列ごとの欠測率・分布の差
report.correlation_table()                   # 変数ペアごとの latent 相関の差
```

`generate` の主な引数:

| 引数 | 意味 |
|---|---|
| `n` | 行数（既定は profile した行数） |
| `seed` | 乱数の seed。同じ profile・`n`・`seed` なら同じ DataFrame になる（同じ numpy のバージョンの場合） |
| `missing` | 欠測の付け方。`"exact"`（既定、各列ちょうど `round(n × 欠測率)` 行）、`"bernoulli"`（行ごとに独立）、`"none"`（欠測なし） |
| `level_map` | 仮名化した level を実名に戻す対応表（後述） |

出力の列順は元の DataFrame と同じです。free text 列と除外した列は出力しません。ID 列は `ID000001` 形式（整数 ID なら 1, 2, …）の新しい値になります。整数列や bool 列に欠測があれば `Int64` / `boolean` になり、Categorical の列は Categorical に戻ります。

`compare(profile, dummy).check()` は次の既定の許容誤差で判定します（`kageframe.DEFAULT_TOLERANCES`、`check(tol={...})` で変更できます）。

- **列**: 同じ列が同じ順に並んでいること。
- **周辺分布**: grid 上の CDF の差が 0.02 以下。平均の差が 0.05 SD 以下。SD の比の差が 5% 以下（裾の重い列は IQR で 10% 以下）。
- **カテゴリ頻度**: level ごとの差が 0.015 以下、TVD が 0.03 以下。profile にない値がないこと。
- **欠測率**: 二項分布の標準誤差の 3 倍以内。
- **相関**: 連続量どうしは 0.03 以下、binary/ordinal を含むペアは 0.05 以下、nominal を含むペアは 0.07 以下。ただし推定量の標準誤差の 4 倍まではペアごとに許容する。全体の RMSE は 0.02 以下。

周辺分布の許容誤差は dummy が 10,000 行の場合の値で、行数 n が少ないと `sqrt(10000 / n)` 倍に広がります。

## 型の指定

型は自動で推定します。推定結果は `kageframe.infer_types(df)` で確認できます。

| 型 | 例 | 自動推定の主な規則 |
|---|---|---|
| `numeric` | 年齢、BMI、検査値 | 数値列（下の条件に当たらないもの） |
| `binary` | 性別、0/1 フラグ、bool | 値が 2 種類の文字列、{0, 1}、bool |
| `ordinal` | ECOG、病期 | 値が 10 種類以下の整数、順序付き Categorical |
| `nominal` | 紹介理由、施設 | 値が 50 種類以下の文字列 |
| `date` / `datetime` / `time` | 検査日、入院日時、来院時刻 | datetime64、日付・時刻の文字列、`datetime.time`、timedelta |
| `id` | 患者 ID | ほぼ全行で異なる短い文字列、連番の整数 |
| `text` | 自由記載 | 長い文字列、種類が多すぎる文字列 |

推定が意図と違う場合や、整数コードの列（1/2/3 など）は `types=` で明示してください。

```python
types = {
    "stage": {"type": "ordinal", "levels": ["I", "II", "III", "IV"]},  # 文字列の順序は推定しない
    "ecog": "ordinal",
    "sex": {"type": "binary", "levels": ["F", "M"]},                    # [陰性側, 陽性側]
    "memo": "excluded",                                                  # 出力から落とす
}
profile = kageframe.profile_dataframe(df, types=types)
```

推定に関する注意は `TypeInferenceWarning` として出ます。警告の内容は `profile.warnings` にも記録されます。

## level 名の仮名化（opt-in）

文字列カテゴリの level 名は、既定ではそのまま profile に保存します。施設名・医師名など、level 名そのものを持ち出したくない列は明示的に仮名化してください。

```python
profile = kageframe.profile_dataframe(df, pseudonymize=["hospital", "doctor"])
# 列ごとに指定することもできる: types={"hospital": {"type": "nominal", "pseudonymize": True}}

profile.save("profile.json")                           # level は L01, L02, ... になっている
profile.save_level_map("level_map.local.json")         # 仮名 → 実名の対応表（持ち出さない）

dummy = kageframe.generate(profile, seed=0)                                     # 仮名のまま
dummy_real = kageframe.generate(profile, seed=0, level_map="level_map.local.json")  # 実名に戻す
kageframe.compare(profile, dummy_real, level_map="level_map.local.json").check()
```

- 仮名は、nominal/binary では頻度の多い順に、ordinal では順序どおりに振ります。
- 整数コードと bool の列は仮名化しません（指定しても警告を出して無視します）。
- 仮名化するかどうかは kageframe が自動では決めません。仮名化していない文字列カテゴリ列が次のどちらかに当たる場合は、`PrivacyWarning`（code `consider_pseudonymize`）で検討を促します。
  - 列名が識別子・人・施設を示す典型パターンに一致する（id, name, hospital, doctor, facility, site, center, 病院, 医師 など）。
  - 値のほぼすべて（90% 以上）が行ごとに異なる。
- 対応表は profile JSON には入りません。`*.local.json` は `.gitignore` 済みです。

## profile に入るもの・入らないもの

profile は、列ごとの集計値と相関行列だけを持つ JSON です。持ち出す前に中身を確認してください（普通のテキストファイルです）。

**入るもの**

- 列名、列の順序、型、元の dtype、非欠測数、欠測率。
- numeric: 平均、SD、要約分位点、再構成用の quantile grid、小数の桁数。値の種類が少ない整数列は、値ごとの確率。
- binary / ordinal / nominal: level 名（または仮名）と件数。
- date / datetime / time: 日数・秒数の quantile grid、時刻の分解能、タイムゾーン名、文字列の書式。
- 列の値が 1 種類だけの場合、その値。
- 変数間の latent 相関行列、ペアごとの同時観測数、profile 中に出た警告。

**入らないもの**

- 行単位のデータ。1 行分の値の組み合わせも含めて一切入りません。
- ID 列の値と書式、free text 列の内容（列名と理由だけを記録します）。
- 真の最小値・最大値。`min`/`max` と grid の端点には、`k/n` 分位と `1 − k/n` 分位を入れます。grid の点数も n/k 以下に抑えるので、各区間の背後に平均 k 行以上があります（既定 `k = 10`、`min_tail_count`）。
- 件数が `k` 未満の nominal level の名前。これらは `Other` にまとめ、まとめた数だけを残します。`Other` 自体が `k` 未満なら、最頻の level に吸収します（`rare_threshold`、既定 10）。binary / ordinal の少ない level はそのまま残り、`PrivacyWarning` が出ます。
- 仮名化した level の実名（対応表は別ファイル）。
- 作成日時・ホスト名などのメタデータ。同じ DataFrame からは常に同じ JSON ができます。

profile は匿名化を保証するものではありません。level 名、件数の少ないカテゴリ、極端な値の近くの分位点から情報が推測される可能性は残ります。持ち出しの可否は所属機関の規則に従って判断してください。

## 仕組みと既知の制限

- 依存構造は Gaussian copula で表します。各列を正規スコアに変換し、latent 相関を推定します。binary/ordinal の組み合わせは Mehler 展開を逆算して補正します。nominal は Gumbel-max で生成するので、level の頻度は正確に再現されます。
- datetime は日付部分と時刻部分に分けて扱います。曜日×時刻のような非単調な依存は再現しません。
- 欠測は MCAR（完全にランダム）で付けます。欠測どうしの相関や、値に依存する欠測は再現しません。
- 日付の前後関係などの論理制約（退院日 ≥ 入院日など）は保証しません。
- 裾の極端な値は、privacy のために `k/n` 分位までしか再現しません。
- 乱数の再現性は、同じ numpy のバージョンの範囲でのみ保証します。

## テスト

`tests/` の主な内容:

| ファイル | 内容 |
|---|---|
| `test_mvp_acceptance.py` | MVP の完成条件 7 項目（列、周辺分布、カテゴリ頻度、欠測率、依存、nominal が 1 つだけ、seed の再現性）を、臨床風 fixture で end-to-end に確認する |
| `test_compare.py` | 自分自身から生成した dummy が通ること。列の欠落・分布のずれ・未知の level・欠測率の変化・依存の破壊を注入すると検出されること |
| `test_missing.py` | 欠測率の再現（exact / bernoulli / none）、MCAR であること、nullable dtype |
| `test_temporal.py` | date / datetime / time の profile（k ルール、分解能、書式、タイムゾーン）と生成 |
| `test_marginals.py`、`test_dependence.py`、`test_psd.py`、`test_generate.py` | 周辺分布、相関推定（真値の回復を含む）、PSD 補正、型ごとの生成 |
| `test_types.py`、`test_schema.py` | 型推定、profile JSON の検証と往復 |

## ライセンス

MIT License です。全文は [LICENSE](LICENSE) を参照してください。Copyright (c) 2026 mizuy
