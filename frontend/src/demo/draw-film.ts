import { beatPulse, phaseAt, shotAt, type SceneName } from './sequence';

type C = CanvasRenderingContext2D;
const W = 960, H = 640;
const rand = (n: number) => { const value = Math.sin(n * 127.1 + 311.7) * 43758.5453; return value - Math.floor(value); };
function fill(c: C, color: string, x = 0, y = 0, w = W, h = H) { c.fillStyle = color; c.fillRect(x, y, w, h); }
function line(c: C, points: number[][], color: string, width = 1) {
  c.beginPath(); points.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.strokeStyle = color; c.lineWidth = width; c.stroke();
}
function poly(c: C, points: number[][], color: string) {
  c.beginPath(); points.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.closePath(); c.fillStyle = color; c.fill();
}
function circle(c: C, x: number, y: number, radius: number, color: string) { c.beginPath(); c.arc(x, y, Math.max(0.01, radius), 0, Math.PI * 2); c.fillStyle = color; c.fill(); }
function gradient(c: C, top: string, bottom: string) { const g = c.createLinearGradient(0, 0, W * 0.25, H); g.addColorStop(0, top); g.addColorStop(1, bottom); c.fillStyle = g; c.fillRect(0, 0, W, H); }
function glow(c: C, x: number, y: number, radius: number, color: string) {
  const g = c.createRadialGradient(x, y, 0, x, y, radius); g.addColorStop(0, color); g.addColorStop(1, 'transparent'); c.fillStyle = g; c.fillRect(x - radius, y - radius, radius * 2, radius * 2);
}
function stars(c: C, time: number, amount = 65) {
  for (let i = 0; i < amount; i++) circle(c, rand(i + 20) * W, rand(i + 900) * 380, 0.5 + rand(i + 200) * 1.2, `rgba(193,220,241,${0.2 + rand(i) * 0.4 + Math.sin(time + i) * 0.05})`);
}
function text(c: C, value: string, x: number, y: number, size: number, color = '#edf3ef', weight = 600) {
  c.font = `${weight} ${size}px 'DM Sans', system-ui, sans-serif`; c.fillStyle = color; c.fillText(value, x, y);
}

function city(c: C, time: number) {
  gradient(c, '#10152e', '#123840'); stars(c, time); glow(c, 480, 315, 390, '#168d8433');
  circle(c, 744, 129, 38, '#d0d7cc'); circle(c, 732, 117, 38, '#161f34');
  for (let layer = 0; layer < 3; layer++) {
    for (let i = 0; i < 22; i++) {
      const x = i * 48 - 30 + layer * 21, h = 50 + rand(i + layer * 48) * (120 + layer * 45), y = 392 - h;
      fill(c, ['#1c3049', '#122639', '#0c1b2c'][layer], x, y, 37 + rand(i) * 17, h);
      if (layer === 2) for (let row = 0; row < h / 15 - 1; row++) for (let col = 0; col < 4; col++) {
        if (rand(i * 71 + row * 6 + col) > 0.48) fill(c, rand(i + col) > 0.5 ? '#57a49c66' : '#dc987055', x + col * 10 + 4, y + row * 15 + 8, 3, 6);
      }
      if (i % 5 === 0) { line(c, [[x + 16, y], [x + 16, y - 28]], '#647e86', 1); circle(c, x + 16, y - 30, 2, '#d19e73'); }
    }
  }
  fill(c, '#0b1824', 0, 393, W, 247);
  poly(c, [[430, 387], [530, 387], [790, H], [170, H]], '#152636');
  line(c, [[430, 390], [170, H]], '#5ae1cd', 2); line(c, [[530, 390], [790, H]], '#71b3bd', 2);
  for (let i = 0; i < 14; i++) {
    const p = ((i / 14 + time * 0.035) % 1) ** 2;
    const y = 398 + p * 290, spread = p * 240;
    line(c, [[477, y], [476 - spread * 0.035, y + 4 + p * 14]], '#d6cda899', 2 + p * 2);
    line(c, [[423 - spread, y], [392 - spread * 1.2, y]], '#3e8d9366', 1 + p);
    line(c, [[536 + spread, y], [568 + spread * 1.2, y]], '#b4806e55', 1 + p);
  }
  for (let side of [-1, 1]) for (let i = 1; i < 7; i++) {
    const p = i / 7, x = 480 + side * (65 + p ** 2 * 430), y = 399 + p ** 2 * 218, height = 18 + p * 115;
    line(c, [[x, y], [x, y - height], [x - side * 20 * p, y - height]], '#476778', 2);
    glow(c, x - side * 19 * p, y - height, 15 + p * 15, '#efb97265');
  }
  // A small, original vehicle silhouette gives the wide shot a focal point.
  const vehicleX = 490 + Math.sin(time * 0.2) * 8;
  poly(c, [[vehicleX - 29, 484], [vehicleX - 22, 466], [vehicleX + 21, 466], [vehicleX + 31, 484], [vehicleX + 31, 494], [vehicleX - 29, 494]], '#08111b');
  fill(c, '#c47256', vehicleX - 24, 482, 10, 3); fill(c, '#c47256', vehicleX + 14, 482, 10, 3);
  glow(c, vehicleX, 496, 43, '#ad685033');
}

function wheel(c: C, time: number) {
  gradient(c, '#152338', '#0b3035'); glow(c, 480, 310, 330, '#39758e44');
  c.save(); c.translate(480, 312); c.rotate(-0.25 + time * 0.45);
  for (let r of [224, 212, 190, 67]) { c.beginPath(); c.arc(0, 0, r, 0, Math.PI * 2); c.strokeStyle = r === 212 ? '#88d4c7' : '#364f62'; c.lineWidth = r === 224 ? 14 : 3; c.stroke(); }
  for (let i = 0; i < 16; i++) {
    const a = i * Math.PI / 8;
    line(c, [[Math.cos(a) * 61, Math.sin(a) * 61], [Math.cos(a + 0.17) * 190, Math.sin(a + 0.17) * 190]], i % 2 ? '#608f9d' : '#b5c7c6', 3);
  }
  circle(c, 0, 0, 45, '#233c4e'); circle(c, 0, 0, 13, '#daaa7b');
  c.restore();
  line(c, [[0, 539], [960, 539]], '#5e8587', 2);
  for (let i = 0; i < 22; i++) { const x = (rand(i) * 1200 - time * 100 + 10000) % 1200 - 120; line(c, [[x, 510 + rand(i + 6) * 40], [x + 40 + rand(i + 90) * 160, 510 + rand(i + 6) * 40]], '#85b3b933', 1); }
}

function tunnel(c: C, time: number) {
  gradient(c, '#19162d', '#102e39'); glow(c, 480, 320, 190, '#29978755');
  for (let i = 16; i >= 0; i--) {
    const z = ((i + time * 2.2) % 17) / 17;
    const r = 18 + z ** 2 * 690;
    const points = Array.from({ length: 7 }, (_, j) => [480 + Math.cos(j * Math.PI / 3 + Math.PI / 6) * r, 310 + Math.sin(j * Math.PI / 3 + Math.PI / 6) * r * 0.82]);
    line(c, points, i % 3 === 0 ? '#cc866c88' : '#4dc4b178', 1 + z * 4);
  }
  for (let i = 0; i < 6; i++) { const a = i * Math.PI / 3 + Math.PI / 6; line(c, [[480 + Math.cos(a) * 24, 310 + Math.sin(a) * 24], [480 + Math.cos(a) * 900, 310 + Math.sin(a) * 750]], '#2b6b7755', 2); }
  circle(c, 480, 310, 15, '#79b9b3');
}

function eye(c: C, time: number) {
  gradient(c, '#212640', '#132c3a'); glow(c, 480, 304, 360, '#8c749633');
  c.save(); c.translate(480, 305);
  c.beginPath(); c.moveTo(-335, 0); c.bezierCurveTo(-150, -195, 175, -180, 335, 0); c.bezierCurveTo(135, 180, -170, 165, -335, 0); c.fillStyle = '#c2c5ba'; c.fill(); c.clip();
  circle(c, 0, 0, 137, '#123545');
  for (let i = 0; i < 180; i++) {
    const a = i / 180 * Math.PI * 2 + time * 0.035, inner = 49 + rand(i) * 18, outer = 110 + rand(i + 44) * 23;
    line(c, [[Math.cos(a) * inner, Math.sin(a) * inner], [Math.cos(a + 0.024) * outer, Math.sin(a + 0.024) * outer]], i % 3 ? '#559fa2' : '#d4a16c', 1.4);
  }
  circle(c, 0, 0, 49 + Math.sin(time * 1.5) * 4, '#071220'); circle(c, -30, -36, 16, '#dce4d8c9'); circle(c, 21, 31, 6, '#e5d8bb66');
  c.restore();
  c.strokeStyle = '#7393a277'; c.lineWidth = 1;
  for (let i = 0; i < 3; i++) { c.beginPath(); c.ellipse(480, 305, 359 + i * 22, 195 + i * 16, -0.04, Math.PI * 1.04, Math.PI * 1.97); c.stroke(); }
}

function wave(c: C, time: number) {
  gradient(c, '#251c3d', '#0e303d'); glow(c, 660, 290, 400, '#234f8055');
  for (let row = 30; row >= 0; row--) {
    const points = Array.from({ length: 90 }, (_, i) => {
      const x = i / 89 * 1100 - 70;
      const y = 170 + row * 11 + Math.sin(x * 0.007 + row * 0.12 + time * 1.3) * (40 + row * 1.7) + Math.sin(x * 0.014 - time * 0.8) * 24;
      return [x, y];
    });
    line(c, points, row % 5 === 0 ? '#ce9a9399' : '#63b5ba75', row % 5 === 0 ? 1.7 : 0.8);
  }
  for (let i = 0; i < 35; i++) circle(c, rand(i + 15) * W, 110 + rand(i + 84) * 400, 1.5, '#98c4d577');
}

function mountain(c: C, time: number) {
  gradient(c, '#39314b', '#aa7f6a'); glow(c, 617, 264, 230, '#edba8755'); circle(c, 618, 264, 70, '#d3ab86');
  for (let layer = 0; layer < 5; layer++) {
    const points: number[][] = [[-50, H]];
    for (let i = 0; i < 14; i++) points.push([i * 86 - 70, 230 + layer * 64 - rand(i + layer * 25) * 145 + Math.sin(time * 0.15) * (5 - layer)]);
    points.push([1040, H]);
    poly(c, points, ['#79647a', '#575d78', '#364d66', '#203c50', '#102838'][layer]);
    if (layer === 1) for (let i = 2; i < points.length - 2; i += 3) { const [x, y] = points[i]; poly(c, [[x, y], [x + 45, y + 70], [x + 19, y + 56], [x - 12, y + 67]], '#91a5b366'); }
  }
  for (let i = 0; i < 7; i++) { c.fillStyle = '#b7c7c514'; c.beginPath(); c.ellipse(120 + i * 150 + Math.sin(time * 0.2) * 10, 430 + i % 3 * 37, 180, 16, 0, 0, Math.PI * 2); c.fill(); }
}

function orbit(c: C, time: number) {
  gradient(c, '#141b36', '#142d3d'); stars(c, time, 110); glow(c, 510, 310, 310, '#397e9144');
  c.save(); c.translate(495, 308); c.rotate(-0.28);
  c.strokeStyle = '#cba58488'; c.lineWidth = 12; c.beginPath(); c.ellipse(0, 0, 340, 79, 0, Math.PI, Math.PI * 2); c.stroke();
  const g = c.createRadialGradient(-70, -65, 5, 35, 55, 228); g.addColorStop(0, '#7bbab5'); g.addColorStop(0.52, '#336b7b'); g.addColorStop(1, '#091628');
  c.fillStyle = g; c.beginPath(); c.arc(0, 0, 168, 0, Math.PI * 2); c.fill();
  c.save(); c.beginPath(); c.arc(0, 0, 166, 0, Math.PI * 2); c.clip();
  for (let i = 0; i < 18; i++) { c.strokeStyle = '#86b5b025'; c.lineWidth = 3; c.beginPath(); c.ellipse(-70 + Math.sin(time * 0.2) * 8, -140 + i * 18, 200, 19, 0.2, 0, Math.PI * 2); c.stroke(); }
  c.restore();
  c.strokeStyle = '#cba584'; c.lineWidth = 7; c.beginPath(); c.ellipse(0, 0, 340, 79, 0, 0, Math.PI); c.stroke();
  const a = time * 0.45; circle(c, Math.cos(a) * 340, Math.sin(a) * 79, 7, '#e8c99e');
  c.restore();
}

const drawScene: Record<SceneName, (c: C, time: number) => void> = { city, wheel, tunnel, eye, wave, mountain, orbit };
const sceneLabels: Record<SceneName, string> = { city: 'NIGHT DRIVE', wheel: 'IN MOTION', tunnel: 'NEW PERSPECTIVE', eye: 'A CLOSER LOOK', wave: 'FEEL THE RHYTHM', mountain: 'THE PAYOFF', orbit: 'GO FURTHER' };

export function drawFilm(c: C, time: number, reducedMotion: boolean) {
  c.save(); c.scale(c.canvas.width / W, c.canvas.height / H);
  const phase = phaseAt(time), shot = shotAt(time), optimized = phase === 'optimized';
  const scene = shot?.scene ?? (phase === 'outro' ? 'mountain' : 'city');
  c.save();
  if (optimized && shot && !reducedMotion) {
    const progress = (time - shot.start) / (shot.end - shot.start), zoom = 1.04 + progress * 0.14;
    c.translate(W / 2, H / 2); c.scale(zoom, zoom); c.rotate(Math.sin(shot.start) * progress * 0.025); c.translate(-W / 2, -H / 2);
  }
  drawScene[scene](c, reducedMotion ? 2 : optimized ? time * 1.3 : time * 0.18);
  c.restore();
  const vignette = c.createRadialGradient(W / 2, H * 0.42, 150, W / 2, H / 2, 580); vignette.addColorStop(0, '#00000000'); vignette.addColorStop(1, '#030b20b0'); c.fillStyle = vignette; c.fillRect(0, 0, W, H);
  fill(c, '#06112055', 0, 0, W, 38); fill(c, '#07132055', 0, 586, W, 54);
  text(c, 'AFTER HOURS / AN ORIGINAL MOTION STORY', 30, 25, 11, '#b2c8ce', 500);
  text(c, phase === 'original' ? 'TAKE 01 — WIDE / CONTINUOUS' : optimized ? `RE-EDIT / ${sceneLabels[scene]}` : 'VIRALBRAIN SI / CONCEPT FILM', 30, 616, 12, '#afc2ca', 500);
  text(c, `${String(Math.floor(time)).padStart(2, '0')}:${String(Math.floor(time % 1 * 30)).padStart(2, '0')}`, 866, 616, 13, '#b2c8ce', 500);
  if (optimized && shot) {
    const progress = (time - shot.start) / (shot.end - shot.start);
    fill(c, '#70c6b388', 30, 566, 900 * progress, 2);
    text(c, shot.caption, 35, 532, 39, '#edf0e8', 650);
    // A small local pulse marks the beat without flashing the full frame.
    circle(c, 912, 66, 3 + beatPulse(time) * 5, '#d4ba86');
  }
  if (phase === 'original') { text(c, 'A city. A road. A possibility.', 35, 546, 27, '#c8d3d0', 450); }
  if (phase === 'intro' || phase === 'transition' || phase === 'outro') {
    fill(c, phase === 'transition' ? '#07111ee0' : '#07111ea9', 0, 38, W, 548);
    const lines = phase === 'intro' ? ['One story.', 'Two edits.'] : phase === 'transition' ? ['Now change', 'the rhythm.'] : ['Edit with', 'intent.'];
    text(c, phase === 'intro' ? 'THE EDITING EXPERIMENT' : phase === 'transition' ? 'VIRALBRAIN SI / CONCEPT RE-EDIT' : 'VIRALBRAIN SI', 57, 178, 15, '#ceb087', 500);
    text(c, lines[0], 52, 292, 78, '#f0f0e9', 550);
    text(c, lines[1], 52, 382, 78, '#99c9c3', 550);
    text(c, phase === 'intro' ? 'Watch what timing, movement and sound can change.' : phase === 'transition' ? 'New cuts. New perspectives. One shared beat.' : 'An editing concept. Ready to explore.', 57, 453, 20, '#b0bdc9', 400);
  }
  c.restore();
}
