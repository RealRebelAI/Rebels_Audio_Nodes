import { app } from "/scripts/app.js";

/*
    Rebels Audio Node Styling
    - Purple section separators between effect groups
    - Collapsible: params hide when section toggle is OFF, show when ON
    - Matches the Gemini mockup image
*/

const SECTIONS = [
    { label: "TRIM",         toggle: "trim",     params: ["trim_start_sec", "trim_end_sec"] },
    { label: "PITCH SHIFT",  toggle: "pitch",    params: ["pitch_semitones"] },
    { label: "TIME STRETCH", toggle: "tempo",    params: ["tempo_rate", "tempo_preserve_pitch"] },
    { label: "FILTER",       toggle: "filter",   params: ["filter_type", "filter_cutoff_hz", "filter_q"] },
    { label: "EQ",           toggle: "eq",       params: ["eq_low_hz", "eq_low_db", "eq_mid_hz", "eq_mid_db", "eq_mid_q", "eq_high_hz", "eq_high_db"] },
    { label: "COMPRESSOR",   toggle: "compress", params: ["compress_threshold", "compress_ratio", "compress_attack_ms", "compress_release_ms", "compress_makeup_db"] },
    { label: "REVERB",       toggle: "reverb",   params: ["reverb_room", "reverb_damping", "reverb_wet", "reverb_dry", "reverb_width"] },
    { label: "CHORUS",       toggle: "chorus",   params: ["chorus_rate_hz", "chorus_depth", "chorus_delay_ms", "chorus_mix"] },
    { label: "GAIN",         toggle: null,       params: [] },
    { label: "FADE",         toggle: null,       params: [] },
    { label: "OUTPUT",       toggle: null,       params: [] },
];

// ─── Widget visibility ───────────────────────────────────────────────────────
function hideWidget(w) {
    if (w._rebelHidden) return;
    w._rebelHidden   = true;
    w._origType      = w.type;
    w._origCompute   = w.computeSize;
    w.type           = "hidden";
    w.computeSize    = () => [0, -4];
}

function showWidget(w) {
    if (!w._rebelHidden) return;
    w._rebelHidden = false;
    w.type         = w._origType;
    w.computeSize  = w._origCompute;
}

// ─── Separator widget (purely visual, no interaction) ─────────────────────────
function makeSeparator(label, collapsible) {
    return {
        type:  "rebel_sep",
        name:  `__sep_${label.toLowerCase().replace(/ /g, "_")}`,
        value: label,
        draw(ctx, node, width, y, height) {
            const pad   = 10;
            const midY  = y + height / 2;
            const color = "#a78bfa";
            const dim   = "#3a3a4a";

            // Dark background band
            ctx.fillStyle = "#1e1e2e";
            ctx.fillRect(0, y, width, height);

            // Purple left accent bar
            ctx.fillStyle = color;
            ctx.fillRect(pad, y + 4, 3, height - 8);

            // Label text
            ctx.fillStyle = color;
            ctx.font      = "bold 11px sans-serif";
            ctx.fillText(label, pad + 9, midY + 4);

            // Trailing line
            const tw = ctx.measureText(label).width;
            ctx.strokeStyle = dim;
            ctx.lineWidth   = 1;
            ctx.beginPath();
            ctx.moveTo(pad + 14 + tw, midY);
            ctx.lineTo(width - (collapsible ? pad + 14 : pad), midY);
            ctx.stroke();

            // Chevron hint for collapsible
            if (collapsible) {
                ctx.fillStyle = "#666";
                ctx.font      = "10px sans-serif";
                ctx.fillText("▾", width - pad - 8, midY + 4);
            }
        },
        computeSize(width) { return [width, 24]; },
        mouse()            { return false; },
        serializeValue()   { return undefined; },
    };
}

// ─── Extension ────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "Rebels.AudioSections",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RebelAudioEdit") return;

        const _onCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (_onCreated) _onCreated.call(this);
            const node = this;

            // Map widget names to widget objects
            const byName = {};
            for (const w of node.widgets) byName[w.name] = w;

            // Insert separators in reverse order to preserve indices
            for (const { label, toggle } of [...SECTIONS].reverse()) {
                const anchorName =
                    toggle        ? toggle
                  : label === "GAIN"   ? "gain_db"
                  : label === "FADE"   ? "fade_in_sec"
                  :                      "preview";

                const anchor = byName[anchorName];
                if (!anchor) continue;
                const idx = node.widgets.indexOf(anchor);
                if (idx === -1) continue;

                node.widgets.splice(idx, 0, makeSeparator(label, !!toggle));
            }

            // Rebuild name map after separators inserted
            const wMap = {};
            for (const w of node.widgets) wMap[w.name] = w;

            // Wire each toggle to collapse/expand its params
            for (const { toggle, params } of SECTIONS) {
                if (!toggle) continue;
                const tw = wMap[toggle];
                if (!tw) continue;

                const pw = params.map(n => wMap[n]).filter(Boolean);

                // Initial state: all toggles default false = collapsed
                const on = tw.value === true;
                for (const p of pw) on ? showWidget(p) : hideWidget(p);

                // On change: show/hide and resize
                const orig = tw.callback;
                tw.callback = function (v) {
                    if (orig) orig.call(this, v);
                    for (const p of pw) v ? showWidget(p) : hideWidget(p);
                    node.setSize(node.computeSize());
                    node.graph?.setDirtyCanvas(true, true);
                };
            }

            node.setSize(node.computeSize());
        };
    },
});
