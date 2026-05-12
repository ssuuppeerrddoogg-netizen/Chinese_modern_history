import re

def extract_time(text):
    patterns = [
        r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日',
        r'(?P<year>\d{4})年(?P<month>\d{1,2})月',
        r'(?P<year>\d{4})年',
        r'(?P<month>\d{1,2})月(?P<day>\d{1,2})日',
        r'(?P<day>\d{1,2})日'
    ]

    times = []

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            g = match.groupdict()

            y = int(g.get('year', 0)) if g.get('year') else 0
            m = int(g.get('month', 0)) if g.get('month') else 0
            d = int(g.get('day', 0)) if g.get('day') else 0

            # 处理年份缺失情况
            if y == 0 and m != 0:
                pre = text[:match.start()]
                y_match = re.search(r'(\d{4})年', pre)
                y = int(y_match.group(1)) if y_match else 0

            # 处理年份和月份缺失情况
            if y == 0 and m == 0 and d != 0:
                pre = text[:match.start()]
                md = re.search(r'(\d{1,2})月(\d{1,2})日', pre)
                if md:
                    m, d = int(md.group(1)), int(md.group(2))
                    y_match = re.search(r'(\d{4})年', pre[:md.start()])
                    y = int(y_match.group(1)) if y_match else 0

            times.append(y * 10000 + m * 100 + d)

    return max(times) if times else None

def extract_event_sentence_with_query(text, query):
    sentences = re.split(r'[。！？]', text)

    best_sentence = None
    best_score = -1

    for s in sentences:
        if not s.strip():
            continue

        score = 0

        # 简单关键词匹配
        for q in query:
            if q in s:
                score += 1

        # 必须包含时间才考虑
        if re.search(r'\d{4}年|\d{1,2}月|\d{1,2}日', s):
            if score > best_score:
                best_score = score
                best_sentence = s

    return best_sentence if best_sentence else sentences[0]

def build_timeline_context(docs, query):
    timeline = []

    for doc in docs:
        text = doc.page_content
        #只取关键句
        event = extract_event_sentence_with_query(text, query)
        t = extract_time(event)
        timeline.append({"t": t, "text": text})

    # 排序
    timeline_sorted = sorted(
        timeline,
        key=lambda x: x["t"] if x["t"] else 99999999
    )

    context = ""
    for item in timeline_sorted:
        t = item["t"]
        text = item["text"]

        if t:
            y, md = t // 10000, t % 10000
            m, d = md // 100, md % 100

            if m == 0 and d == 0:
                context += f"[{y}年] {text}\n\n"
            elif d == 0:
                context += f"[{y}年{m}月] {text}\n\n"
            else:
                context += f"[{y}年{m}月{d}日] {text}\n\n"
        else:
            context += text + "\n\n"

    return context, timeline_sorted