"""Draw the CliffWalking board and learned policy."""

from __future__ import annotations

import cv2
import numpy as np

from .env import CliffWalkingEnv


COLORS = {
    "background": (247, 249, 252),
    "grid": (205, 213, 224),
    "safe": (255, 255, 255),
    "cliff": (63, 73, 191),
    "start": (91, 171, 91),
    "goal": (73, 190, 235),
    "text": (48, 57, 70),
}


def _center_text(
    image: np.ndarray,
    text: str,
    center: tuple[int, int],
    color: tuple[int, int, int] = COLORS["text"],
    scale: float = 0.5,
) -> None:
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    point = (center[0] - width // 2, center[1] + (height - baseline) // 2)
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_board(
    env: CliffWalkingEnv,
    q_table: np.ndarray | None = None,
    cell: int = 72,
    agent_position: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return an image of the board, optionally showing the greedy policy."""
    pad, header = 24, 48
    image = np.full(
        (header + env.rows * cell + pad, env.cols * cell + 2 * pad, 3),
        COLORS["background"],
        dtype=np.uint8,
    )
    cv2.putText(image, "CliffWalking", (pad, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLORS["text"], 2, cv2.LINE_AA)

    for row in range(env.rows):
        for col in range(env.cols):
            position = (row, col)
            x1, y1 = pad + col * cell, header + row * cell
            x2, y2 = x1 + cell, y1 + cell
            center = (x1 + cell // 2, y1 + cell // 2)
            color = (
                COLORS["start"] if position == env.start else
                COLORS["goal"] if position == env.goal else
                COLORS["cliff"] if position in env.cliff else COLORS["safe"]
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)

            if position in env.cliff:
                for is_outer, start, end in (
                    ((row - 1, col) not in env.cliff, (x1, y1), (x2, y1)),
                    ((row + 1, col) not in env.cliff, (x1, y2), (x2, y2)),
                    ((row, col - 1) not in env.cliff, (x1, y1), (x1, y2)),
                    ((row, col + 1) not in env.cliff, (x2, y1), (x2, y2)),
                ):
                    if is_outer:
                        cv2.line(image, start, end, COLORS["grid"], 1)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), COLORS["grid"], 1)

            if position == env.start:
                _center_text(image, "START", center)
            elif position == env.goal:
                _center_text(image, "GOAL", center)
            elif q_table is not None:
                action = int(np.argmax(q_table[env.state_id(position)]))
                dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[action]
                end = (center[0] + dx * cell // 4, center[1] + dy * cell // 4)
                cv2.arrowedLine(image, center, end, COLORS["text"], 2, cv2.LINE_AA)

            if agent_position == position:
                cv2.circle(image, center, cell // 5, (48, 105, 235), cv2.FILLED)
                cv2.circle(image, center, cell // 5, (255, 255, 255), 2)

    left = pad + cell
    right = pad + (env.cols - 1) * cell
    center = ((left + right) // 2, header + (env.rows - 1) * cell + cell // 2)
    _center_text(image, "CLIFF", center, (255, 255, 255), 0.55)
    return image
