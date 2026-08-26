"""Reporting helpers: tables, figures, JSON, and a small markdown writer."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_tqdm(iterable, **kw):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kw)
    except Exception:
        return iterable


def setup_bangla_font(candidates=None):
    """Try to enable a Bangla-capable font for figure labels.

    Returns the font name if found, else None (caller should rely on CSVs for
    Bangla text, since matplotlib's default fonts cannot render Bengali).
    """
    import matplotlib as mpl
    import matplotlib.font_manager as fm

    if not candidates:
        candidates = [
            "Noto Sans Bengali", "Noto Serif Bengali", "Kalpurush", "Siyam Rupali",
            "Mukti", "AdorshoLipi", "Lohit Bengali", "Nikosh", "Vrinda", "Shonar Bangla",
        ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            mpl.rcParams["font.family"] = c
            mpl.rcParams["axes.unicode_minus"] = False
            return c
    return None


def save_table(df, path, index=True):
    path = Path(path)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def df_to_markdown(df, index=True):
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


class ReportWriter:
    """Accumulate markdown and write a report.md into the reports dir."""

    def __init__(self, out_dir, title):
        self.out = ensure_dir(out_dir)
        self.lines = [f"# {title}", ""]

    def h2(self, text):
        self.lines += [f"## {text}", ""]
        return self

    def h3(self, text):
        self.lines += [f"### {text}", ""]
        return self

    def p(self, text):
        self.lines += [text, ""]
        return self

    def bullet(self, text):
        self.lines.append(f"- {text}")
        return self

    def gap(self):
        self.lines.append("")
        return self

    def table(self, df, index=True):
        self.lines += [df_to_markdown(df, index=index), ""]
        return self

    def image(self, rel_path, caption=""):
        self.lines += [f"![{caption}]({rel_path})", ""]
        if caption:
            self.lines += [f"*{caption}*", ""]
        return self

    def write(self, name="report.md"):
        path = self.out / name
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"[report] wrote {path}")
        return path
