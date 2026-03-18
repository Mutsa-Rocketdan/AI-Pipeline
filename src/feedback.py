"""정답/오답 피드백 및 취약 영역 분석 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class QuizResult:
    quiz_id: int
    quiz_type: str
    topic: str
    difficulty: str
    source_date: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str = ""


@dataclass
class SessionStats:
    results: list[QuizResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.results if r.is_correct)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct_count / self.total * 100

    @property
    def score(self) -> int:
        if self.total == 0:
            return 0
        return round(self.correct_count / self.total * 100)

    def accuracy_by_type(self) -> dict[str, dict]:
        """퀴즈 유형별 정답률."""
        type_stats: dict[str, dict] = {}
        for r in self.results:
            if r.quiz_type not in type_stats:
                type_stats[r.quiz_type] = {"total": 0, "correct": 0}
            type_stats[r.quiz_type]["total"] += 1
            if r.is_correct:
                type_stats[r.quiz_type]["correct"] += 1

        for stats in type_stats.values():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
        return type_stats

    def accuracy_by_difficulty(self) -> dict[str, dict]:
        """난이도별 정답률."""
        diff_stats: dict[str, dict] = {}
        for r in self.results:
            d = r.difficulty
            if d not in diff_stats:
                diff_stats[d] = {"total": 0, "correct": 0}
            diff_stats[d]["total"] += 1
            if r.is_correct:
                diff_stats[d]["correct"] += 1

        for stats in diff_stats.values():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
        return diff_stats

    def accuracy_by_date(self) -> dict[str, dict]:
        """강의 날짜별 정답률."""
        date_stats: dict[str, dict] = {}
        for r in self.results:
            d = r.source_date
            if d not in date_stats:
                date_stats[d] = {"total": 0, "correct": 0}
            date_stats[d]["total"] += 1
            if r.is_correct:
                date_stats[d]["correct"] += 1

        for stats in date_stats.values():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
        return date_stats

    def get_weak_areas(self) -> list[dict]:
        """취약 영역 분석: 정답률이 낮은 주제/날짜를 반환."""
        date_stats = self.accuracy_by_date()
        weak = []
        for date_key, stats in date_stats.items():
            if stats["accuracy"] < 60 and stats["total"] >= 2:
                weak.append({
                    "date": date_key,
                    "accuracy": stats["accuracy"],
                    "total": stats["total"],
                    "correct": stats["correct"],
                })
        weak.sort(key=lambda x: x["accuracy"])
        return weak

    def get_wrong_questions(self) -> list[QuizResult]:
        """오답 문항 리스트."""
        return [r for r in self.results if not r.is_correct]

    def get_recommendations(self) -> list[str]:
        """학습 추천 메시지 생성."""
        recommendations = []
        weak_areas = self.get_weak_areas()

        if not self.results:
            return ["아직 퀴즈를 풀지 않았습니다. 퀴즈를 풀어보세요!"]

        if self.accuracy >= 90:
            recommendations.append("훌륭합니다! 전반적으로 학습 내용을 잘 이해하고 있습니다.")
            recommendations.append("더 높은 난이도에 도전해 보세요.")
        elif self.accuracy >= 70:
            recommendations.append("좋은 성적입니다. 틀린 문제의 해설을 다시 확인해 보세요.")
        elif self.accuracy >= 50:
            recommendations.append("강의 내용을 다시 복습하는 것을 권장합니다.")
        else:
            recommendations.append("기초 개념부터 다시 학습하는 것이 필요합니다.")

        for area in weak_areas[:3]:
            recommendations.append(
                f"{area['date']} 강의 내용을 집중적으로 복습하세요. "
                f"(정답률: {area['accuracy']:.0f}%)"
            )

        type_stats = self.accuracy_by_type()
        for qtype, stats in type_stats.items():
            if stats["accuracy"] < 50 and stats["total"] >= 2:
                type_names = {
                    "multiple_choice": "객관식",
                    "short_answer": "주관식",
                    "fill_blank": "빈칸 채우기",
                    "code": "코드",
                }
                name = type_names.get(qtype, qtype)
                recommendations.append(f"{name} 유형 문제를 더 연습해 보세요.")

        return recommendations


def check_answer(user_answer: str, correct_answer: str, quiz_type: str = "multiple_choice") -> bool:
    """사용자 답변의 정답 여부 확인."""
    user_clean = user_answer.strip().lower()
    correct_clean = correct_answer.strip().lower()

    def normalize(text: str) -> str:
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_items(text: str) -> list[str]:
        """
        정답이 여러 항목인 경우를 감지하기 위한 분리 함수.
        예: "버퍼, 채널, 셀렉터" -> ["버퍼", "채널", "셀렉터"]
        """
        text = normalize(text)
        text = text.replace(" 및 ", ",").replace(" 그리고 ", ",").replace(" & ", ",")
        parts = re.split(r"[,/;|·]+", text)
        items = [p.strip() for p in parts if p.strip()]
        # 중복 제거(순서 유지)
        deduped = []
        seen = set()
        for item in items:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    if not user_clean:
        return False

    if quiz_type == "multiple_choice":
        return user_clean == correct_clean

    if quiz_type in ("short_answer", "fill_blank"):
        user_clean = normalize(user_clean)
        correct_clean = normalize(correct_clean)

        # 정답이 여러 항목(쉼표 등)으로 구성된 경우: 모든 항목을 포함해야 정답 처리
        expected_items = split_items(correct_clean)
        if len(expected_items) >= 2:
            return all(item in user_clean for item in expected_items)

        if user_clean == correct_clean:
            return True
        # 단일 항목 정답일 때만 포함 관계 허용
        if correct_clean in user_clean or user_clean in correct_clean:
            return True
        user_words = set(user_clean.replace(",", " ").replace(".", " ").split())
        correct_words = set(correct_clean.replace(",", " ").replace(".", " ").split())
        if correct_words and len(user_words & correct_words) / len(correct_words) >= 0.7:
            return True

    return user_clean == correct_clean
