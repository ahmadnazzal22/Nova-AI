import re
from typing import Set, Dict, List
from collections import Counter

# ── Arabic ↔ English mixed-language pairs ──────────────────────────
_ARABIC_EQUIVALENTS: Dict[str, str] = {
    # Tech / programming
    "ai": "الذكاء الاصطناعي",
    "artificial intelligence": "الذكاء الاصطناعي",
    "machine learning": "التعلم الآلي",
    "deep learning": "التعلم العميق",
    "neural network": "الشبكة العصبية",
    "algorithm": "الخوارزمية",
    "algorithms": "الخوارزميات",
    "transformer": "المحول",
    "transformers": "المحولات",
    "attention": "الانتباه",
    "self-attention": "الانتباه الذاتي",
    "embedding": "التضمين",
    "embeddings": "التضمينات",
    "token": "الرمز",
    "tokens": "الرموز",
    "dataset": "مجموعة البيانات",
    "training": "التدريب",
    "inference": "الاستدلال",
    "optimizer": "المحسّن",
    "loss function": "دالة الخسارة",
    "accuracy": "الدقة",
    "precision": "الضبط",
    "recall": "الاستدعاء",
    "layer": "الطبقة",
    "layers": "الطبقات",
    "encoder": "المشفر",
    "decoder": "المفكك",
    "fine-tuning": "الضبط الدقيق",
    "pretrained": "مدرب مسبقاً",
    "overfitting": "الإفراط في التكيف",
    "underfitting": "ضعف التكيف",
    "batch": "الدفعة",
    "epoch": "الدورة",
    "gradient": "التدرج",
    "backpropagation": "الانتشار العكسي",
    "activation function": "دالة التنشيط",
    "softmax": "سوفتماكس",
    "vector": "المتجه",
    "vectors": "المتجهات",
    "dimension": "البعد",
    "dimensionality": "الأبعاد",
    # General computing
    "server": "الخادم",
    "database": "قاعدة البيانات",
    "api": "واجهة البرمجة",
    "cloud": "السحابة",
    "framework": "الإطار",
    "library": "المكتبة",
    "deployment": "النشر",
    "pipeline": "خط الأنابيب",
}

_INTENT_PATTERNS: Dict[str, List[str]] = {
    "research": [
        "research", "بحث", "study", "دراسة", "overview", "نظرة عامة",
        "comprehensive", "شامل", "survey", "استقصاء", "literature",
    ],
    "explanation": [
        "explain", "شرح", "what is", "ما هو", "what are", "ما هي",
        "how does", "كيف", "define", "عرف", "describe", "صف",
        "tell me about", "حدثني عن", "understanding", "فهم",
        "why", "لماذا", "why is", "why does", "why do", "why are",
    ],
    "comparison": [
        "compare", "قارن", "difference", "الفرق", "versus", "مقابل",
        "vs", "better", "أفضل", "pros and cons", "المزايا والعيوب",
        "similarities", "أوجه التشابه",
    ],
    "list": [
        "list", "اذكر", "list of", "قائمة", "types of", "أنواع",
        "examples of", "أمثلة", " enumerate", "عدد", "top", "أفضل",
    ],
    "steps": [
        "steps", "خطوات", "step by step", "خطوة بخطوة", "how to",
        "كيفية", "guide", "دليل", "tutorial", "تعليمي", "process",
        "عملية", "procedure", "إجراء", "instructions", "تعليمات",
    ],
    "summary": [
        "summary", "ملخص", "summarize", "لخص", "brief", "موجز",
        "recap", "خلاصة", "tl;dr", "in short", "باختصار",
    ],
}

_RESEARCH_HEADINGS = [
    "Introduction",
    "Definition",
    "Main Concepts",
    "Detailed Explanation",
    "Steps",
    "Applications",
    "Advantages",
    "Challenges",
    "Conclusion",
]

_RESEARCH_HEADINGS_AR = [
    "المقدمة",
    "التعريف",
    "المفاهيم الرئيسية",
    "الشرح التفصيلي",
    "الخطوات",
    "التطبيقات",
    "المزايا",
    "التحديات",
    "الخاتمة",
]

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

_PARAGRAPH_MAX_SENTENCES = 5


class ResponseFormatter:
    """Formats LLM responses into clean, professional, ChatGPT-quality output."""

    def __init__(self):
        self._intent_cache: Dict[str, str] = {}

    # ── Public API ──────────────────────────────────────────────────

    def format(self, answer: str, question: str = "", detect_intent: bool = True) -> str:
        if not answer:
            return answer

        text = answer.strip()
        import re
        text = re.sub(r'^#{1,6}\s*.+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        intent = self._detect_intent(question) if detect_intent and question else "general"

        # 0. If sports/match/news with uncertain RAG knowledge, return Arabic apology
        if self._is_unknown_sports_or_news(question, text):
            return "عذراً، لا أملك معلومات كافية حول هذا الموضوع. يرجى البحث في مصادر متخصصة."

        # 1. Remove question echo
        text = self._remove_question_echo(text, question)

        # 2. Remove decorative separators
        text = self._remove_decorative_separators(text)

        # 3. Remove duplicated titles and paragraphs
        text = self._deduplicate_all(text)

        # 4. Clean mixed-language (Arabic context)
        text = self._clean_mixed_language(text)

        # 5. Normalize heading spacing
        text = self._normalize_heading_spacing(text)

        # 7. Enforce paragraph length
        text = self._enforce_paragraph_length(text)

        # 8. Split inline numbered lists (1. 2. 3. on same line)
        text = self._split_inline_lists(text)

        # 9. Ensure list spacing
        text = self._ensure_list_spacing(text)

        # 10. Collapse excessive blank lines (max 1)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 11. If research intent and poorly structured, restructure
        if intent == "research" and self._is_poorly_formatted(text):
            text = self._restructure_as_research(text, question)

        return text.strip()

    # ── Intent Detection ────────────────────────────────────────────

    def _detect_intent(self, question: str) -> str:
        if not question:
            return "general"
        q = question.lower().strip()
        if q in self._intent_cache:
            return self._intent_cache[q]

        scores: Dict[str, int] = {k: 0 for k in _INTENT_PATTERNS}
        for intent, patterns in _INTENT_PATTERNS.items():
            for pat in patterns:
                if pat in q:
                    scores[intent] += 1

        # Prefer specific intents over general
        ordered = ["research", "steps", "comparison", "list", "summary", "explanation"]
        best = "general"
        best_score = 0
        for intent in ordered:
            if scores[intent] > best_score:
                best = intent
                best_score = scores[intent]

        self._intent_cache[q] = best
        return best

    # ── Language Detection ──────────────────────────────────────────

    def _is_arabic(self, text: str) -> bool:
        matches = _ARABIC_RANGE.findall(text)
        return len(matches) > 5  # enough Arabic chars to consider it Arabic

    # ── Mixed Language Cleanup ──────────────────────────────────────

    def _clean_mixed_language(self, text: str) -> str:
        if not self._is_arabic(text):
            return text

        result = text
        # Sort by length (longest first) to avoid partial replacements
        for eng, arb in sorted(_ARABIC_EQUIVALENTS.items(), key=lambda x: -len(x[0])):
            # Case-insensitive replacement of standalone English terms
            pattern = re.compile(rf"\b{re.escape(eng)}\b", re.IGNORECASE)
            result = pattern.sub(arb, result)
        return result

    # ── Question Echo Removal ───────────────────────────────────────

    def _remove_question_echo(self, text: str, question: str) -> str:
        lines = text.split("\n")
        q_lower = question.strip().lower()
        filtered: List[str] = []
        seen_echo = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered.append(line)
                continue
            lower = stripped.lower().rstrip(".?!")
            # Check if line starts with **<question>** (bold echo)
            stripped_no_bold = stripped.replace("**", "").replace("*", "").strip()
            is_bold_echo = (
                stripped.startswith("**")
                and stripped_no_bold.lower().startswith(q_lower[:30].rstrip(".?!"))
            )
            is_echo = (
                lower == q_lower.rstrip(".?!")
                or lower.startswith(q_lower[:40].rstrip(".?!"))
                or lower.startswith("you asked about")
                or lower.startswith("you asked:")
                or lower.startswith("question:")
                or lower.startswith("q:")
                or stripped.startswith('"') and stripped.endswith('"') and lower.strip('"').strip() == q_lower.rstrip(".?!")
                or lower.startswith("here is")
                or lower.startswith("here's")
                or lower.startswith("sure")
                or lower.startswith("certainly")
                or lower.startswith("of course")
                or is_bold_echo
            )
            if not is_echo or seen_echo:
                filtered.append(line)
                if is_echo:
                    seen_echo = True
            else:
                seen_echo = True
        return "\n".join(filtered).strip()

    # ── Sports/News Detection ───────────────────────────────────────

    _SPORTS_NEWS_KEYWORDS = re.compile(
        r"(sport|match|score|league|tournament|champion|"
        r"team|player|coach|stadium|goal|win|lose|"
        r"news|breaking|update|live|result|"
        r"game|cup|final|quarter|playoff|"
        r"رياضة|مباراة|نتيجة|دوري|كأس|هداف|فريق|لاعب)", re.I
    )

    _UNCERTAIN_ANSWER_PATTERNS = re.compile(
        r"(i don'?t have|i don'?t know|i'm not sure|i am not sure|"
        r"i cannot|unable to provide|no information|"
        r"not in my knowledge|not in my training|"
        r"لا أعرف|لا أملك|لا توجد معلومات)", re.I
    )

    def _is_unknown_sports_or_news(self, question: str, answer: str) -> bool:
        if not question or not answer:
            return False
        has_sports_keywords = bool(self._SPORTS_NEWS_KEYWORDS.search(question))
        has_uncertainty = bool(self._UNCERTAIN_ANSWER_PATTERNS.search(answer))
        is_short_answer = len(answer.strip().split()) < 15
        return has_sports_keywords and (has_uncertainty or is_short_answer)

    # ── Decorative Separators ───────────────────────────────────────

    _DECORATIVE_PATTERN = re.compile(r"^\s*[=\-*_]{4,}\s*$", re.MULTILINE)

    def _remove_decorative_separators(self, text: str) -> str:
        return self._DECORATIVE_PATTERN.sub("", text).strip()

    # ── Deduplication ───────────────────────────────────────────────

    _TRANSITION_PHRASES = [
        "in conclusion", "to summarize", "to sum up", "in summary",
        "finally", "lastly", "in short", "overall",
    ]

    def _deduplicate_all(self, text: str) -> str:
        """Multi-pass dedup: headings, sentences, paragraphs, bigrams, transitions."""
        text = self._dedup_headings(text)
        text = self._dedup_sentences(text)
        text = self._dedup_paragraphs(text)
        text = self._dedup_bigram_overlap(text)
        text = self._remove_redundant_transitions(text)
        return text

    def _dedup_headings(self, text: str) -> str:
        seen: Set[str] = set()
        lines = text.split("\n")
        result: List[str] = []
        for line in lines:
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
            if heading_match:
                key = heading_match.group(2).strip().lower()
                if key in seen:
                    continue
                seen.add(key)
            result.append(line)
        return "\n".join(result)

    def _dedup_sentences(self, text: str) -> str:
        """Remove near-duplicate sentences across the entire text."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen: Set[int] = set()
        result: List[str] = []
        for sent in sentences:
            stripped = sent.strip()
            if not stripped:
                result.append(sent)
                continue
            # Normalize: lowercase, strip punctuation, collapse whitespace
            norm = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", "", stripped.lower())
            norm = re.sub(r"\s+", " ", norm).strip()
            # Use first 60 chars as key (catches slight rewording)
            key = hash(norm[:60])
            if key not in seen:
                seen.add(key)
                result.append(sent)
        return " ".join(result)

    def _dedup_paragraphs(self, text: str) -> str:
        paragraphs = re.split(r"\n\s*\n", text)
        seen_keys: Set[int] = set()
        result: List[str] = []
        for para in paragraphs:
            normalized = para.strip().lower()
            if not normalized:
                result.append(para)
                continue
            # Compare on first 150 chars
            key = hash(normalized[:150])
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(para)
        return "\n\n".join(result)

    def _dedup_bigram_overlap(self, text: str) -> str:
        """Merge consecutive paragraphs with high bigram overlap (>60%)."""
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) < 2:
            return text
        result = [paragraphs[0]]
        for i in range(1, len(paragraphs)):
            prev = result[-1].strip().lower()
            curr = paragraphs[i].strip().lower()
            if not prev or not curr:
                result.append(paragraphs[i])
                continue
            prev_bigrams = set(zip(prev.split(), prev.split()[1:]))
            curr_bigrams = set(zip(curr.split(), curr.split()[1:]))
            if not prev_bigrams or not curr_bigrams:
                result.append(paragraphs[i])
                continue
            overlap = len(prev_bigrams & curr_bigrams) / min(len(prev_bigrams), len(curr_bigrams))
            if overlap > 0.6:
                continue  # skip near-duplicate paragraph
            result.append(paragraphs[i])
        return "\n\n".join(result)

    def _remove_redundant_transitions(self, text: str) -> str:
        """Remove redundant transition phrases that appear more than once."""
        lower = text.lower()
        counts = Counter()
        for phrase in self._TRANSITION_PHRASES:
            counts[phrase] = lower.count(phrase)
        # Keep only the first occurrence of each overused transition
        result_lines = []
        seen_transitions: Set[str] = set()
        for line in text.split("\n"):
            lower_line = line.strip().lower()
            matched = None
            for phrase in self._TRANSITION_PHRASES:
                if lower_line.startswith(phrase) or lower_line.startswith(f"**{phrase}**"):
                    matched = phrase
                    break
            if matched and counts[matched] > 1:
                if matched not in seen_transitions:
                    seen_transitions.add(matched)
                    result_lines.append(line)
                # else skip — already emitted one
            else:
                result_lines.append(line)
        return "\n".join(result_lines)

    # ── Heading Spacing ─────────────────────────────────────────────

    def _normalize_heading_spacing(self, text: str) -> str:
        text = re.sub(r"(?<=\S)\n(#{1,6}\s+)", r"\n\n\1", text)
        # Ensure newline after heading before content
        text = re.sub(r"(#{1,6}\s+.+)\n(?=\S)", r"\1\n", text)
        return text

    # ── Paragraph Length Enforcement ────────────────────────────────

    def _enforce_paragraph_length(self, text: str) -> str:
        """Split paragraphs that exceed _PARAGRAPH_MAX_SENTENCES sentences."""
        paragraphs = re.split(r"\n\s*\n", text) if "\n\n" in text else [text]
        result: List[str] = []
        for para in paragraphs:
            stripped = para.strip()
            if not stripped or stripped.startswith("#"):
                result.append(para)
                continue
            # Count sentences
            sentences = re.split(r"(?<=[.!?])\s+", stripped)
            if len(sentences) <= _PARAGRAPH_MAX_SENTENCES:
                result.append(para)
            else:
                # Split into chunks of _PARAGRAPH_MAX_SENTENCES
                for i in range(0, len(sentences), _PARAGRAPH_MAX_SENTENCES):
                    chunk = " ".join(sentences[i:i + _PARAGRAPH_MAX_SENTENCES])
                    result.append(chunk)
        return "\n\n".join(result)

    # ── Inline List Splitting ───────────────────────────────────────

    _INLINE_LIST_RE = re.compile(r"(?<=\d\.\s)(?=\*\*?[^\*]+\*\*?|\w+(?::| – | — ))")

    def _split_inline_lists(self, text: str) -> str:
        """Split inline list items (numbered 1. or bullet * / -) onto separate lines."""
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            # Count numbered items (1. 2. 3.) — split if >1
            num_items = re.findall(r"\d+\.\s", stripped)
            if len(num_items) > 1:
                parts = re.split(r"\s+(?=\d+\.\s)", stripped)
                result.append("\n".join(parts))
                continue
            # Count inline bullet items (* text * text) — split if >1
            bullet_match = re.findall(r"(?:^|\s)\*\s", stripped)
            if len(bullet_match) > 1:
                bullet_parts = re.split(r"\s+(?=\*\s)", stripped)
                if len(bullet_parts) > 1:
                    result.append("\n".join(bullet_parts))
                    continue
            result.append(line)
        return "\n".join(result)

    # ── List Spacing ────────────────────────────────────────────────

    def _ensure_list_spacing(self, text: str) -> str:
        text = re.sub(r"(?<=\S)\n(?:[-*+] |\d+\.\s)", r"\n\n\g<0>", text)
        return text

    # ── Research Restructuring ──────────────────────────────────────

    def _is_poorly_formatted(self, text: str) -> bool:
        """Check if the text lacks proper structure (headings, sections)."""
        has_headings = bool(re.search(r"^##+\s+", text, re.MULTILINE))
        has_sections = bool(re.search(r"\n\s*\n", text))
        return not (has_headings and has_sections)

    def _restructure_as_research(self, text: str, question: str) -> str:
        """Rebuild text into the standard research template."""
        is_arabic = self._is_arabic(text)
        headings = _RESEARCH_HEADINGS_AR if is_arabic else _RESEARCH_HEADINGS

        # Extract title from question
        title = question.strip().rstrip(".?!")
        title = title[:80]

        # Divide text into roughly equal sections
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        # Filter out headings
        content_paras = [p for p in paragraphs if not re.match(r"^#{1,6}\s+", p)]

        # Assign content to sections
        sections = {h: [] for h in headings}
        if is_arabic:
            # Intro first, conclusion last, distribute rest
            num_content = len(content_paras)
            section_keys = list(sections.keys())
            for i, para in enumerate(content_paras):
                if i == 0:
                    sections[section_keys[0]].append(para)
                elif i == num_content - 1:
                    sections[section_keys[-1]].append(para)
                else:
                    idx = 1 + ((i - 1) * (len(section_keys) - 2) // max(1, num_content - 2))
                    idx = min(idx, len(section_keys) - 2)
                    sections[section_keys[idx]].append(para)
        else:
            num_content = len(content_paras)
            section_keys = list(sections.keys())
            for i, para in enumerate(content_paras):
                if i == 0:
                    sections[section_keys[0]].append(para)
                elif i == num_content - 1:
                    sections[section_keys[-1]].append(para)
                else:
                    idx = 1 + ((i - 1) * (len(section_keys) - 2) // max(1, num_content - 2))
                    idx = min(idx, len(section_keys) - 2)
                    sections[section_keys[idx]].append(para)

        # Build final output
        lines: List[str] = []
        if is_arabic:
            lines.append(f"# {title}")
        else:
            lines.append(f"# {title}")
        lines.append("")

        for heading in headings:
            content = sections.get(heading, [])
            if not content:
                continue
            lines.append(f"## {heading}")
            lines.append("")
            for para in content:
                lines.append(para)
                lines.append("")

        return "\n".join(lines).strip()


# ── Module-level convenience ────────────────────────────────────────

_formatter = ResponseFormatter()


def format_response(answer: str, question: str = "", detect_intent: bool = True) -> str:
    return _formatter.format(answer, question, detect_intent)
