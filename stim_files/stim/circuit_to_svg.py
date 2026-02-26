"""
Generate SVG diagrams from a Stim circuit file.

Outputs:
  <stem>_timeline.svg          — full gate timeline (Stim built-in)
  <stem>_layout_tick{N}.svg    — spatial layout at tick N (Stim built-in, b&w)
  <stem>_detectors_tick{N}.svg — detector regions at tick N (Stim built-in)
  <stem>_qubit_layout.svg      — colored surface-code layout with q-labels (custom)

Usage:
    python circuit_to_svg.py [circuit_file] [--tick N] [--timeline-only]

Defaults:
    circuit_file = circuit_ideal.stim
    tick         = 0
"""
import argparse
import pathlib
from collections import defaultdict

import stim


# ── helpers ───────────────────────────────────────────────────────────────────

def save_svg(content: str, path: pathlib.Path) -> None:
    path.write_text(str(content), encoding="utf-8")
    print(f"Saved: {path}")


# ── circuit parsing ───────────────────────────────────────────────────────────

def parse_circuit_layout(circuit: stim.Circuit):
    """
    Walk the circuit (including REPEAT blocks) and return:
      coords       : {qubit_index -> (x, y)}
      qubit_types  : {qubit_index -> 'data' | 'x_ancilla' | 'z_ancilla'}
      connections  : list of (gate_name, q1, q2)  — unique gate interactions
    """
    coords: dict[int, tuple[float, float]] = {}
    measured: set[int] = set()
    h_seen:   set[int] = set()
    connections: list[tuple[str, int, int]] = []

    def visit(instructions):
        for inst in instructions:
            name = inst.name

            if name == "QUBIT_COORDS":
                q = inst.targets_copy()[0].value
                c = inst.gate_args_copy()
                coords[q] = (float(c[0]), float(c[1]))

            elif name in {"M", "MR", "MX", "MY", "MRX", "MRY"}:
                for t in inst.targets_copy():
                    if not t.is_measurement_record_target:
                        measured.add(t.value)

            elif name == "H":
                for t in inst.targets_copy():
                    h_seen.add(t.value)

            elif name in {"CZ", "CX", "CNOT"}:
                tgts = inst.targets_copy()
                for i in range(0, len(tgts) - 1, 2):
                    q1 = tgts[i].value
                    q2 = tgts[i + 1].value
                    connections.append((name, q1, q2))

            elif name == "REPEAT":
                visit(inst.body_copy())

    visit(circuit)

    qubit_types: dict[int, str] = {}
    for q in coords:
        if q in measured:
            qubit_types[q] = "x_ancilla" if q in h_seen else "z_ancilla"
        else:
            qubit_types[q] = "data"

    return coords, qubit_types, connections


# ── custom colored SVG layout ─────────────────────────────────────────────────

# Colors: fill, stroke
_COLORS = {
    "data":      ("#FFFFFF", "#333333"),
    "x_ancilla": ("#FF6B6B", "#CC0000"),
    "z_ancilla": ("#4A90D9", "#0055AA"),
    "unknown":   ("#AAAAAA", "#555555"),
}


def _xml(*parts: str) -> str:
    return "\n".join(parts)


def make_qubit_layout_svg(
    coords: dict,
    qubit_types: dict,
    connections: list,
    title: str = "Surface Code — Qubit Layout",
    scale: int = 80,
    pad: int = 55,
) -> str:
    """
    Build a standalone SVG string showing the 2-D qubit layout with:
      - White circles   → data qubits
      - Red squares     → X-type ancillas
      - Blue squares    → Z-type ancillas
      - q{i} labels on every qubit
      - Gray lines for 2-qubit gate connections
      - A legend and color-coded grid background
    The SVG uses a viewBox so it scales freely in any browser.
    """
    if not coords:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'

    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def px(x, y):
        """Map circuit coordinates → SVG pixel coordinates."""
        return (pad + (x - min_x) * scale,
                pad + (y - min_y) * scale)

    canvas_w = 2 * pad + (max_x - min_x) * scale
    canvas_h = 2 * pad + (max_y - min_y) * scale
    legend_h  = 90
    total_h   = canvas_h + legend_h

    r = scale * 0.28          # half-size of qubit markers (radius or half-side)
    font_sz = max(7, r * 0.65)

    lines: list[str] = []

    # ── SVG header ────────────────────────────────────────────────────────────
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 0 {canvas_w:.1f} {total_h:.1f}"'
        f' width="{canvas_w:.0f}" height="{total_h:.0f}">'
    )

    # Background
    lines.append(f'<rect width="{canvas_w:.1f}" height="{total_h:.1f}" fill="#F0F4F8"/>')

    # Subtle grid aligned to coordinate grid
    for gx in range(int(min_x), int(max_x) + 1):
        sx, _ = px(gx, min_y)
        _, ey = px(gx, max_y)
        lines.append(f'<line x1="{sx:.1f}" y1="{pad/2:.1f}" x2="{sx:.1f}" y2="{canvas_h - pad/2:.1f}"'
                     f' stroke="#CCCCCC" stroke-width="0.5"/>')
    for gy in range(int(min_y), int(max_y) + 1):
        _, sy = px(min_x, gy)
        ex, _ = px(max_x, gy)
        lines.append(f'<line x1="{pad/2:.1f}" y1="{sy:.1f}" x2="{canvas_w - pad/2:.1f}" y2="{sy:.1f}"'
                     f' stroke="#CCCCCC" stroke-width="0.5"/>')

    # Title
    lines.append(
        f'<text x="{canvas_w / 2:.1f}" y="22"'
        f' text-anchor="middle" font-family="monospace" font-size="14"'
        f' font-weight="bold" fill="#222222">{title}</text>'
    )

    # ── connections (drawn behind qubits) ─────────────────────────────────────
    drawn_edges: set[frozenset] = set()
    for gate, q1, q2 in connections:
        key = frozenset({q1, q2})
        if key in drawn_edges or q1 not in coords or q2 not in coords:
            continue
        drawn_edges.add(key)
        x1, y1 = px(*coords[q1])
        x2, y2 = px(*coords[q2])
        stroke = "#777777" if gate == "CZ" else "#AAAAAA"
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{stroke}" stroke-width="2" opacity="0.75"/>'
        )

    # ── qubit markers + labels ────────────────────────────────────────────────
    for q in sorted(coords):
        cx, cy = px(*coords[q])
        qtype  = qubit_types.get(q, "unknown")
        fill, stroke = _COLORS[qtype]

        if qtype == "data":
            lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
                f' fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
        else:
            lines.append(
                f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}"'
                f' width="{2 * r:.1f}" height="{2 * r:.1f}"'
                f' fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )

        lines.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}"'
            f' text-anchor="middle" dominant-baseline="central"'
            f' font-family="monospace" font-size="{font_sz:.1f}"'
            f' font-weight="bold" fill="#111111">q{q}</text>'
        )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        ("data",      "circle", "Data qubit (circle)"),
        ("x_ancilla", "square", "X-type ancilla (square)"),
        ("z_ancilla", "square", "Z-type ancilla (square)"),
    ]
    lr   = 9    # legend marker half-size
    lx0  = pad
    ly   = canvas_h + 14
    step = (canvas_w - 2 * pad) / len(legend_items)

    lines.append(
        f'<rect x="0" y="{canvas_h:.1f}" width="{canvas_w:.1f}" height="{legend_h:.1f}"'
        f' fill="#E0E8F0"/>'
    )
    lines.append(
        f'<text x="{canvas_w / 2:.1f}" y="{canvas_h + 11:.1f}"'
        f' text-anchor="middle" font-family="monospace" font-size="10"'
        f' fill="#444444">Legend</text>'
    )

    for i, (qtype, shape, label) in enumerate(legend_items):
        fill, stroke = _COLORS[qtype]
        lxi = lx0 + i * step
        lyi = ly + 14
        if shape == "circle":
            lines.append(
                f'<circle cx="{lxi + lr:.1f}" cy="{lyi:.1f}" r="{lr}"'
                f' fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        else:
            lines.append(
                f'<rect x="{lxi:.1f}" y="{lyi - lr:.1f}"'
                f' width="{2 * lr}" height="{2 * lr}"'
                f' fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        lines.append(
            f'<text x="{lxi + 2 * lr + 6:.1f}" y="{lyi:.1f}"'
            f' dominant-baseline="central"'
            f' font-family="monospace" font-size="11" fill="#222222">{label}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert a .stim circuit to SVG diagrams.")
    parser.add_argument(
        "circuit_file",
        nargs="?",
        default="circuit_ideal.stim",
        help="Path to the .stim circuit file (default: circuit_ideal.stim)",
    )
    parser.add_argument(
        "--tick",
        type=int,
        default=0,
        help="Which tick to render for Stim's timeslice diagram (default: 0)",
    )
    parser.add_argument(
        "--timeline-only",
        action="store_true",
        help="Only generate the full timeline SVG (skip all layout diagrams)",
    )
    args = parser.parse_args()

    circuit_path = pathlib.Path(args.circuit_file)
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

    circuit = stim.Circuit.from_file(str(circuit_path))
    stem    = circuit_path.stem

    # ── Stim built-in: full timeline ──────────────────────────────────────────
    save_svg(circuit.diagram("timeline-svg"),
             circuit_path.parent / f"{stem}_timeline.svg")

    if not args.timeline_only:
        # ── Stim built-in: timeslice (b&w, no q-labels) ───────────────────────
        save_svg(circuit.diagram("timeslice-svg", tick=args.tick),
                 circuit_path.parent / f"{stem}_layout_tick{args.tick}.svg")

        # ── Stim built-in: detector regions ───────────────────────────────────
        try:
            save_svg(circuit.diagram("detector-slice-svg", tick=args.tick),
                     circuit_path.parent / f"{stem}_detectors_tick{args.tick}.svg")
        except Exception as e:
            print(f"Skipping detector-slice diagram: {e}")

        # ── Custom: colored layout with q-labels ──────────────────────────────
        coords, qubit_types, connections = parse_circuit_layout(circuit)

        counts = {t: sum(v == t for v in qubit_types.values())
                  for t in ("data", "x_ancilla", "z_ancilla")}
        print(f"Qubits — data: {counts['data']}, "
              f"X-ancilla: {counts['x_ancilla']}, "
              f"Z-ancilla: {counts['z_ancilla']}")

        svg = make_qubit_layout_svg(
            coords, qubit_types, connections,
            title=f"Surface Code Layout — {circuit_path.name}",
        )
        save_svg(svg, circuit_path.parent / f"{stem}_qubit_layout.svg")
        print("Tip: open *_qubit_layout.svg in Chrome/Firefox — Ctrl+scroll to zoom.")


if __name__ == "__main__":
    main()
