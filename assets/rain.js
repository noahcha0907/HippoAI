/* Ripple shader adapted from CFGI-Repo/week3.html (Face Rain).
   One WebGL context, fifteen square viewports. Invisible falling bodies trigger
   the original impact shader; no sphere, shadow, texture, or camera is rendered. */
(() => {
  'use strict';
  const canvas = document.querySelector('#rain');
  if (!canvas) return;
  const status = document.querySelector('#rain-status');
  const pause = document.querySelector('#pause-rain');
  const shuffle = document.querySelector('#shuffle-rain');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
  function unavailable(message) {
    if (status) { status.textContent = message; status.hidden = false; }
    canvas.hidden = true;
    if (pause) pause.disabled = true;
    if (shuffle) shuffle.disabled = true;
  }
  if (!gl) { unavailable('Still waters. Animation is unavailable in this browser.'); return; }
  const vertexSource = `
    attribute vec2 a_position;
    uniform vec2 u_resolution, u_center;
    uniform float u_radius;
    varying vec2 v_position;
    void main() {
      v_position = a_position;
      gl_Position = vec4((u_center + a_position * u_radius) / u_resolution * 2.0 - 1.0, 0.0, 1.0);
    }`;
  const rippleSource = `
    precision mediump float;
    varying vec2 v_position;
    uniform float u_age, u_extent;
    // ShaderCode's RGB phase shifts, 50-frequency radial waves, reciprocal
    // distortion, fract bands, and 9x output gain, evaluated per impact.
    vec3 shaderCode(vec2 st, float age) {
      float rnd = max(st.y, 0.02);
      vec3 c = vec3(0.0);
      float l = 0.0;
      float z = age;
      for (int i = 0; i < 3; i++) {
        st -= vec2(0.5);
        z -= 1.0;
        l = max(length(st), 0.002);
        float radialWave = sin(l * 50.0 - 2.0 * z);
        float divisor = (rnd / l) * radialWave;
        // Safe reciprocal replaces pow(negativeValue, -1.0), undefined in GLSL.
        st *= (divisor < 0.0 ? -1.0 : 1.0) / max(abs(divisor), 0.002);
        float band = fract(0.01 / max(length(st - 0.5), 0.002));
        c[i] = band * 5.0 / (1.0 + l * 3.0);
      }
      return clamp(c * 9.0 + l, 0.0, 1.0);
    }
    void main() {
      vec2 offset = v_position * u_extent;
      float distance = length(offset);
      float arrival = u_age * 85.0;
      // An outward-moving front and trailing rings carry the original pattern.
      float front = 1.0 - smoothstep(arrival, arrival + 30.0, distance);
      float wake = exp(-max(arrival - distance, 0.0) / 190.0);
      float rings = 0.35 + 0.65 * pow(0.5 + 0.5 * cos(distance * 0.07 - u_age * 5.95), 3.0);
      float fade = (1.0 - smoothstep(20.0, 30.0, u_age)) * smoothstep(0.0, 0.15, u_age);
      vec3 effect = shaderCode(offset / max(180.0, arrival * 1.5) + 0.5, u_age);
      gl_FragColor = vec4(mix(vec3(1.0), effect, front * wake * rings * fade * 0.85), 1.0);
    }
  `;
  let program;
  try {
    program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, rippleSource]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader); gl.deleteShader(shader);
    }
    gl.bindAttribLocation(program, 0, 'a_position'); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  } catch (error) { console.error(error); unavailable('Still waters. The animation could not start.'); return; }
  const uniforms = {};
  for (const key of ['resolution', 'center', 'radius', 'age', 'extent']) uniforms[key] = gl.getUniformLocation(program, 'u_' + key);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.useProgram(program); gl.enable(gl.BLEND); gl.blendFunc(gl.DST_COLOR, gl.ZERO);
  gl.clearColor(1, 1, 1, 1);
  const SIZE = 320, STEP = 1 / 120;
  let cells = [], time = 0, accumulator = 0, previous = 0, animation = 0;
  let paused = reduced.matches, visible = true, lost = false;
  function reset() {
    time = 0; accumulator = 0;
    cells = Array.from({ length: 15 }, (_, index) => ({
      ball: null, next: Math.random() * 3,
      ripples: [{ x: 105 + Math.random() * 110, y: 105 + Math.random() * 110, born: -(0.9 + (index % 5) * 0.55 + Math.random()) }]
    }));
    draw();
  }
  function simulate(dt) {
    time += dt;
    for (const cell of cells) {
      if (!cell.ball && time >= cell.next) {
        cell.ball = { x: 105 + Math.random() * 110, y: 105 + Math.random() * 110, z: 330 + Math.random() * 50, vz: -35 };
      }
      if (cell.ball) {
        cell.ball.vz -= 650 * dt;
        cell.ball.z += cell.ball.vz * dt;
        if (cell.ball.z <= 24) {
          cell.ripples.push({ x: cell.ball.x, y: cell.ball.y, born: time });
          cell.ball = null;
          cell.next = time + 3.5 + Math.random() * 3;
        }
      }
      cell.ripples = cell.ripples.filter(r => time - r.born < 30);
    }
  }
  function draw() {
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    // Integer tile size ensures every cell is a square, without seams.
    const side = Math.max(1, Math.round(canvas.clientWidth * dpr / 5));
    if (canvas.width !== side * 5 || canvas.height !== side * 3) {
      canvas.width = side * 5; canvas.height = side * 3;
    }
    gl.disable(gl.SCISSOR_TEST); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST);
    cells.forEach((cell, i) => {
      const x = (i % 5) * side, y = (2 - Math.floor(i / 5)) * side;
      gl.viewport(x, y, side, side); gl.scissor(x, y, side, side);
      gl.uniform2f(uniforms.resolution, SIZE, SIZE);
      for (const ripple of cell.ripples) {
        const age = time - ripple.born, extent = Math.min(age * 85 + 40, Math.hypot(SIZE, SIZE));
        gl.uniform2f(uniforms.center, ripple.x, ripple.y);
        gl.uniform1f(uniforms.radius, extent);
        gl.uniform1f(uniforms.extent, extent); gl.uniform1f(uniforms.age, age);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    });
  }
  function active() { return !paused && !document.hidden && visible && !lost; }
  function frame(now) {
    animation = 0;
    if (!active()) return;
    accumulator += Math.min((now - previous) / 1000, 0.05); previous = now;
    while (accumulator >= STEP) { simulate(STEP); accumulator -= STEP; }
    draw(); animation = requestAnimationFrame(frame);
  }
  function sync() {
    cancelAnimationFrame(animation); animation = 0;
    if (pause) {
      pause.textContent = paused ? '▶ Play motion' : 'Ⅱ Pause motion';
      pause.setAttribute('aria-pressed', String(paused));
    }
    if (active()) { previous = performance.now(); animation = requestAnimationFrame(frame); }
  }
  pause?.addEventListener('click', () => { paused = !paused; sync(); });
  shuffle?.addEventListener('click', () => { reset(); sync(); });
  reduced.addEventListener('change', () => { paused = reduced.matches; sync(); });
  document.addEventListener('visibilitychange', sync);
  const observer = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); });
  observer.observe(canvas);
  const resize = new ResizeObserver(() => { if (!lost) draw(); });
  resize.observe(canvas);
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault(); lost = true; cancelAnimationFrame(animation);
    unavailable('Animation interrupted. Reload to restart.');
  });
  window.addEventListener('pagehide', () => { cancelAnimationFrame(animation); });
  window.addEventListener('pageshow', sync);
  reset(); sync();
})();
