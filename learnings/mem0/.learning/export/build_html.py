#!/usr/bin/env python3
"""Build a shareable single-file HTML course from repo-mastery artifacts.

Reads:
  - ../COVERAGE.md            → course overview / map (module 00)
  - ../chapters/m0X.md        → one HTML module per chapter
  - html-shell/_base.html, _footer.html, styles.css, main.js

Writes (in this dir):
  modules/*.html  → per-module <section> fragments
  index.html      → assembled single-file course
  styles.css, main.js  (copied verbatim from the shell)
"""
import html
import os
import re
import shutil

SHELL = "/Users/zhangyongshun/.claude/skills/repo-mastery/references/html-shell"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACCENT = {"color": "#2C7C9C", "hover": "#225F77", "light": "#E6F1F5", "muted": "#5F9DB8"}
COURSE_TITLE = "Mem0 深度源码课程"


# ---------- tiny markdown → html (subset used in chapters) ----------

def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def _table(rows):
    out = ["<div class=\"table-wrap\"><table>"]
    for row in rows:
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
        tag = "th"
        out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        # code fence
        if line.strip().startswith("```"):
            buf, i = [], i + 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        # table: detect a line with | and next line of ---
        if "|" in line and i + 1 < n and re.match(r"^[\s:\-|]+$", lines[i + 1]):
            rows = []
            while i < n and "|" in lines[i]:
                rows.append(lines[i]); i += 1
            out.append(_table(rows))
            continue
        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            tag = f"h{lvl+1}"
            out.append(f"<{tag}>{_inline(m.group(2))}</{tag}>")
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        if line.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if re.match(r"^>\s*", line):
            buf, i = [], i + 1
            buf.append(_inline(re.sub(r"^>\s*", "", line)))
            while i < n and re.match(r"^>\s*", lines[i]):
                buf.append(_inline(re.sub(r"^>\s*", "", lines[i]))); i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue
        if re.match(r"^[-*]\s+", line):
            buf, i = [], i + 1
            buf.append(_inline(re.sub(r"^[-*]\s+", "", line)))
            while i < n and re.match(r"^[-*]\s+", lines[i]):
                buf.append(_inline(re.sub(r"^[-*]\s+", "", lines[i]))); i += 1
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in buf) + "</ul>")
            continue
        if re.match(r"^\d+\.\s+", line):
            buf, i = [], i + 1
            buf.append(_inline(re.sub(r"^\d+\.\s+", "", line)))
            while i < n and re.match(r"^\d+\.\s+", lines[i]):
                buf.append(_inline(re.sub(r"^\d+\.\s+", "", lines[i]))); i += 1
            out.append("<ol>" + "".join(f"<li>{b}</li>" for b in buf) + "</ol>")
            continue
        # paragraph (merge consecutive non-blank text lines)
        buf, i = [line], i + 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}|\s*>|\s*[-*]\s|\s*\d+\.|```|\s*\|)|^---$", lines[i]):
            buf.append(lines[i]); i += 1
        out.append("<p>" + _inline(" ".join(b.strip() for b in buf)) + "</p>")
    return "\n".join(out)


def module_section(number: str, title: str, subtitle: str, body_html: str) -> str:
    return f"""<section class="module" id="module-{number}">
  <div class="module-content">
    <div class="module-header">
      <span class="module-number">{number}</span>
      <h2 class="module-title">{html.escape(title)}</h2>
      <p class="module-subtitle">{html.escape(subtitle)}</p>
    </div>
    <div class="screen">{body_html}</div>
  </div>
</section>"""


def main():
    # copy shell static assets verbatim
    for f in ("styles.css", "main.js", "_footer.html"):
        shutil.copy(os.path.join(SHELL, f), os.path.join(HERE, f))

    base = open(os.path.join(SHELL, "_base.html")).read()
    footer = open(os.path.join(HERE, "_footer.html")).read()

    # nav dots: one per module (00..07)
    modules_meta = [
        ("00", "课程地图与价值", "Mem0 深度源码课程 · 7 模块 / 23 知识点总览"),
        ("01", "定位与生态", "mem0 是什么、vs 谁、何时选"),
        ("02", "构建与环境", "装起来、配好、用起来"),
        ("03", "总体架构心智模型", "分层 / 存储 / 三入口"),
        ("04", "写入管线 add()", "V3 八阶段 · 抽取 · 去重 · 实体"),
        ("05", "检索管线 search()", "多信号融合 · threshold · 实体加分"),
        ("06", "设计哲学与权衡", "ADD-only · 代价 · 平台分界"),
        ("07", "动手实验与决策链", "demo 实跑 · 自研 vs 集成"),
    ]
    dots = "\n".join(
        f'      <span class="nav-dot" data-target="module-{n}" title="{t}"></span>'
        for n, t, _ in modules_meta
    )

    # module 00 from COVERAGE
    cover = open(os.path.join(ROOT, "COVERAGE.md")).read()
    m00_body = md_to_html(cover)
    m00 = module_section("00", modules_meta[0][1], modules_meta[0][2], m00_body)

    # modules 01..07 from chapters
    mods = [m00]
    for num, title, subtitle in modules_meta[1:]:
        path = os.path.join(ROOT, "chapters", f"m{num}.md")
        body = md_to_html(open(path).read())
        mods.append(module_section(num, title, subtitle, body))

    os.makedirs(os.path.join(HERE, "modules"), exist_ok=True)
    for i, mod in enumerate(mods):
        fn = os.path.join(HERE, "modules", f"{i:02d}-module.html")
        with open(fn, "w") as f:
            f.write(mod + "\n")
        print("wrote", fn)

    # assemble index.html
    base = base.replace("COURSE_TITLE", COURSE_TITLE)
    base = base.replace("NAV_DOTS", dots)
    base = base.replace("ACCENT_COLOR", ACCENT["color"])
    base = base.replace("ACCENT_HOVER", ACCENT["hover"])
    base = base.replace("ACCENT_LIGHT", ACCENT["light"])
    base = base.replace("ACCENT_MUTED", ACCENT["muted"])
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(base + "\n".join(mods) + "\n" + footer)
    print("Built index.html")


if __name__ == "__main__":
    main()
