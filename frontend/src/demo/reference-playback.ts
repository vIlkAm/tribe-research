import { REFERENCE_CLIPS, REFERENCE_SPLIT, referenceDuration, referencePosition, referenceTimeline, referenceVideoRequest } from './reference-clips.ts';
import type { YouTubePlayer } from './youtube.ts';

export const initialPlayback = () => ({ time: 0, videoDuration: 20, playing: false, requested: false, status: 'Preparing YouTube…' });
export type ReferencePlaybackState = ReturnType<typeof initialPlayback>;

/** One transport for both the page controls and YouTube's own controls. */
export class ReferencePlayback {
  state = initialPlayback();
  private player: YouTubePlayer | null = null;
  private index = 0;
  private pending = false;
  private complete = false;
  private hold = false;
  private correctedStart = false;
  private hasPlayed = false;
  private waitingSince = 0;
  private publish: (state: ReferencePlaybackState) => void;

  constructor(publish: (state: ReferencePlaybackState) => void) { this.publish = publish; }
  private update(patch: Partial<ReferencePlaybackState>) {
    this.state = { ...this.state, ...patch };
    this.publish(this.state);
  }
  prepare(time: number) {
    this.player = null; this.index = referencePosition(time, this.state.videoDuration).index;
    this.pending = false; this.complete = false; this.hold = false; this.correctedStart = false;
    this.hasPlayed = false;
    this.update({ time, playing: false, requested: false, status: 'Preparing YouTube…' });
  }
  attach(player: YouTubePlayer) {
    this.player = player;
    // The iframe already has the correct clip and start. Re-cueing here can race
    // with the first click and replace a playing video with its thumbnail.
    this.update({ status: 'Ready · press Play' });
  }
  private matches() {
    const url = this.player?.getVideoUrl();
    if (!url) return false;
    try { return new URL(url).searchParams.get('v') === REFERENCE_CLIPS[this.index].id; }
    catch { return false; }
  }
  private duration() {
    if (this.index !== 1 || !this.matches()) return;
    const seconds = this.player!.getDuration();
    if (Number.isFinite(seconds) && seconds > REFERENCE_CLIPS[1].start && seconds !== this.state.videoDuration) this.update({ videoDuration: seconds });
  }
  go(next: number, run: boolean) {
    const player = this.player;
    if (!player) return;
    const target = referencePosition(next, this.state.videoDuration);
    const time = target.clip.offset + target.sourceTime - target.clip.start;
    const sameClip = this.index === target.index && this.matches();
    const resume = sameClip && !this.complete && Math.abs(time - this.state.time) < 0.05;
    this.index = target.index; this.complete = false; this.hold = !run;
    this.correctedStart = false;
    const needsCue = !sameClip || (!this.hasPlayed && !(resume && run));
    this.pending = needsCue;
    this.waitingSince = run ? Date.now() : 0;
    this.update({ time, requested: run, playing: false, status: run ? 'Starting video…' : 'Paused' });
    if (needsCue) {
      this.hasPlayed = false;
      const request = referenceVideoRequest(target.index, target.sourceTime);
      if (run) player.loadVideoById(request); else player.cueVideoById(request);
    } else if (resume && run) {
      player.playVideo();
    } else {
      player.seekTo(target.sourceTime, true);
      if (run) player.playVideo(); else player.pauseVideo();
    }
  }
  toggle() {
    if (this.state.requested) this.pause();
    else this.go(this.complete ? 0 : this.state.time, true);
  }
  pause(status = 'Paused') {
    this.hold = true; this.waitingSince = 0;
    this.player?.pauseVideo();
    this.update({ requested: false, playing: false, status });
  }
  blocked() { this.pause('Press Play in the YouTube video to continue.'); this.hold = false; }
  private finish() {
    if (this.complete || this.pending) return;
    if (this.index === 0) { this.go(REFERENCE_SPLIT, true); return; }
    this.duration(); this.complete = true; this.waitingSince = 0;
    this.update({ time: referenceDuration(this.state.videoDuration), requested: false, playing: false, status: 'Comparison complete' });
  }
  onState(code: number) {
    if (!this.player || !this.matches()) return;
    if (code === 1) {
      this.pending = false;
      if (this.hold) { this.player.pauseVideo(); return; }
      this.hasPlayed = true;
      this.waitingSince = 0; this.complete = false; this.duration();
      this.update({ requested: true, playing: true, status: 'Playing' });
    } else if (code === 3) {
      this.update({ playing: false, status: this.state.requested ? 'Buffering · brain paused' : 'Paused' });
    } else if (code === 5) {
      this.pending = false;
      if (this.state.requested) this.player.playVideo();
      else { this.hold = false; this.update({ playing: false }); }
    } else if (code === 2) {
      if (this.pending) return;
      this.hold = false;
      this.update({ requested: false, playing: false, status: this.complete ? 'Comparison complete' : 'Paused' });
    } else if (code === 0 && !this.pending && !this.complete) {
      // A stale ended event from the previous clip must not finish the new clip.
      const time = this.player.getCurrentTime();
      const end = this.index === 0 ? REFERENCE_CLIPS[0].end : this.player.getDuration();
      if (end > 0 && time >= end - 0.3) this.finish();
    }
  }
  tick(now = Date.now()) {
    const player = this.player;
    if (!player || !this.matches() || this.complete) return;
    const code = player.getPlayerState();
    // Reconcile a missed message instead of keeping the page stuck in buffering.
    if (code === 1 && !this.hold) {
      if (!this.state.playing || this.pending) this.onState(1);
      const mediaTime = player.getCurrentTime();
      const clip = REFERENCE_CLIPS[this.index];
      if (mediaTime < clip.start - 0.2) {
        if (!this.correctedStart) { this.correctedStart = true; player.seekTo(clip.start, true); }
        return;
      }
      this.duration();
      if (!clip.naturalEnd && mediaTime >= clip.end) this.finish();
      else this.update({ time: referenceTimeline(this.index, mediaTime, this.state.videoDuration) });
    } else if (code === 3 && this.state.playing) this.onState(3);
    else if (code === 2 && !this.pending && (this.hold || this.state.playing)) this.onState(2);
    else if (code === 0 && !this.pending) this.onState(0);
    if (this.waitingSince && now - this.waitingSince > 10000 && code !== 1) {
      this.waitingSince = 0;
      if (code === 3) this.update({ playing: false, status: 'Video is still buffering · brain paused' });
      else {
        this.update({ requested: false, playing: false, status: 'Press Play in the YouTube video to continue.' });
        this.hold = false;
      }
    }
  }
}
