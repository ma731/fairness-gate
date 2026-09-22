"""A WebGL point cloud of the people the model judged.

Why raw WebGL2 rather than Three.js: the dependency is the problem, not the technique.
A CDN script in a compliance artifact is a supply-chain surface and stops the page from
opening offline in five years; vendoring Three.js puts 600KB of someone else's code in a
repository whose whole argument is that everything in it is checked. This is about 4KB
and has no dependencies at all.

What it draws is the data, not an ornament. Every point is a person, sampled in the
exact proportions of the real confusion counts, and the ones the model overlooks are the
ones that glow. If WebGL is unavailable the canvas stays empty and the CSS mesh behind
it carries the hero, so nothing on the page depends on this.

The HTML it emits is deterministic: four integers and a fixed seed. Geometry is computed
in the browser from that seed, so the gate's byte-identical check still holds.
"""

from __future__ import annotations

POINTS = 26000
SEED = 20260922


def totals(result: dict, split: str = "test", attribute: str = "RAC1P") -> dict:
    conf = (
        result.get("distributions", {})
        .get(split, {})
        .get(attribute, {})
        .get("confusion")
        or []
    )
    return {
        "tp": sum(c["tp"] for c in conf),
        "fn": sum(c["fn"] for c in conf),
        "fp": sum(c["fp"] for c in conf),
        "tn": sum(c["tn"] for c in conf),
    }


def markup(result: dict) -> str:
    t = totals(result)
    total = sum(t.values())
    if not total:
        return ""
    return (
        '<canvas class="field" id="field" aria-hidden="true"></canvas>'
        f'<script>window.__FIELD__={{tp:{t["tp"]},fn:{t["fn"]},fp:{t["fp"]},'
        f'tn:{t["tn"]},n:{POINTS},seed:{SEED}}};</script>'
    )


SCRIPT = r"""
(function () {
  var cfg = window.__FIELD__;
  var cv = document.getElementById('field');
  if (!cfg || !cv) return;
  var gl = cv.getContext('webgl2', { antialias: true, alpha: true,
                                     premultipliedAlpha: false });
  // No WebGL: the CSS mesh behind the canvas already carries the hero.
  if (!gl) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  var N = cfg.n, total = cfg.tp + cfg.fn + cfg.fp + cfg.tn;
  // Thresholds in the real proportions, so what you see is the actual composition.
  var cut1 = cfg.tp / total,
      cut2 = cut1 + cfg.fn / total,
      cut3 = cut2 + cfg.fp / total;

  // Seeded so every visitor sees the same field and the page stays reproducible.
  var seed = cfg.seed >>> 0;
  function rnd() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  }

  var pos = new Float32Array(N * 3), cat = new Float32Array(N);
  var GA = Math.PI * (3 - Math.sqrt(5));
  for (var i = 0; i < N; i++) {
    // Fibonacci sphere: even coverage with no clumping, and deterministic.
    var y = 1 - (i / (N - 1)) * 2, r = Math.sqrt(Math.max(0, 1 - y * y)), th = GA * i;
    var jitter = 0.86 + rnd() * 0.3;
    pos[i * 3] = Math.cos(th) * r * jitter;
    pos[i * 3 + 1] = y * jitter;
    pos[i * 3 + 2] = Math.sin(th) * r * jitter;
    var u = rnd();
    cat[i] = u < cut1 ? 0 : u < cut2 ? 1 : u < cut3 ? 2 : 3;
  }

  function sh(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  }
  var vs = sh(gl.VERTEX_SHADER,
    '#version 300 es\n' +
    'in vec3 aPos; in float aCat;' +
    'uniform float uT; uniform vec2 uRes; uniform vec2 uM;' +
    'out float vCat; out float vZ;' +
    'void main(){' +
    ' float a = uT*0.11 + uM.x*0.6; float b = -0.22 + uM.y*0.4;' +
    ' vec3 p = aPos;' +
    ' float ca=cos(a), sa=sin(a);' +
    ' p = vec3(p.x*ca + p.z*sa, p.y, -p.x*sa + p.z*ca);' +
    ' float cb=cos(b), sb=sin(b);' +
    ' p = vec3(p.x, p.y*cb - p.z*sb, p.y*sb + p.z*cb);' +
    ' float z = p.z + 3.05;' +
    ' vec2 q = p.xy / z * 1.75;' +
    ' q.x *= uRes.y / uRes.x;' +
    ' gl_Position = vec4(q, 0.0, 1.0);' +
    ' gl_PointSize = clamp(210.0 / z / 26.0, 1.0, 6.5);' +
    ' vCat = aCat; vZ = z;' +
    '}');
  var fs = sh(gl.FRAGMENT_SHADER,
    '#version 300 es\nprecision highp float;' +
    'in float vCat; in float vZ;' +
    'uniform vec3 cA; uniform vec3 cB; uniform vec3 cC; uniform vec3 cD;' +
    'out vec4 o;' +
    'void main(){' +
    ' vec2 d = gl_PointCoord - 0.5; float r = length(d);' +
    ' if (r > 0.5) discard;' +
    ' float soft = smoothstep(0.5, 0.06, r);' +
    ' vec3 col = cD; float al = 0.10;' +
    ' if (vCat < 0.5) { col = cA; al = 0.30; }' +
    ' else if (vCat < 1.5) { col = cB; al = 0.95; }' +
    ' else if (vCat < 2.5) { col = cC; al = 0.38; }' +
    ' float fade = clamp(1.7 / vZ, 0.18, 1.5);' +
    ' o = vec4(col, al * soft * fade);' +
    '}');
  if (!vs || !fs) return;

  var pr = gl.createProgram();
  gl.attachShader(pr, vs); gl.attachShader(pr, fs); gl.linkProgram(pr);
  if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) return;
  gl.useProgram(pr);

  function buf(data, loc, size) {
    var b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    var l = gl.getAttribLocation(pr, loc);
    gl.enableVertexAttribArray(l);
    gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0);
  }
  buf(pos, 'aPos', 3);
  buf(cat, 'aCat', 1);

  function css(name, fb) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(v || fb);
    var h = m ? m[1] : fb;
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255,
            parseInt(h.slice(4, 6), 16) / 255];
  }
  function colors() {
    gl.uniform3fv(gl.getUniformLocation(pr, 'cA'), css('--s1', '3987e5'));
    gl.uniform3fv(gl.getUniformLocation(pr, 'cB'), css('--acid', 'ccff00'));
    gl.uniform3fv(gl.getUniformLocation(pr, 'cC'), css('--s2', 'd95926'));
    gl.uniform3fv(gl.getUniformLocation(pr, 'cD'), css('--ink', 'f6f8f9'));
  }
  colors();
  new MutationObserver(colors).observe(document.documentElement,
    { attributes: true, attributeFilter: ['data-theme'] });

  var uT = gl.getUniformLocation(pr, 'uT'),
      uRes = gl.getUniformLocation(pr, 'uRes'),
      uM = gl.getUniformLocation(pr, 'uM');

  function size() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = cv.clientWidth, h = cv.clientHeight;
    cv.width = Math.max(1, (w * dpr) | 0);
    cv.height = Math.max(1, (h * dpr) | 0);
    gl.viewport(0, 0, cv.width, cv.height);
    gl.uniform2f(uRes, cv.width, cv.height);
  }
  size();
  addEventListener('resize', size, { passive: true });

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);   // additive: overlapping points glow

  var mx = 0, my = 0, tx = 0, ty = 0;
  addEventListener('pointermove', function (e) {
    tx = (e.clientX / innerWidth - 0.5) * 2;
    ty = (e.clientY / innerHeight - 0.5) * 2;
  }, { passive: true });

  // Stop drawing entirely when the hero is off screen or the tab is hidden. A
  // background animation that keeps burning a GPU behind six other sections is rude.
  var live = true;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (es) { live = es[0].isIntersecting; },
      { threshold: 0.01 }).observe(cv);
  }

  var t0 = performance.now();
  (function frame(now) {
    requestAnimationFrame(frame);
    if (!live || document.hidden) return;
    mx += (tx - mx) * 0.045;
    my += (ty - my) * 0.045;
    gl.uniform1f(uT, (now - t0) / 1000);
    gl.uniform2f(uM, mx, my);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, cfg.n);
  })(t0);
})();
"""
