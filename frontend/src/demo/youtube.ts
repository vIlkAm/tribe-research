interface VideoRequest { videoId: string; startSeconds: number; endSeconds?: number; }
export interface YouTubePlayer {
  loadVideoById(request: VideoRequest): void;
  cueVideoById(request: VideoRequest): void;
  playVideo(): void;
  pauseVideo(): void;
  seekTo(seconds: number, allowSeekAhead: boolean): void;
  mute(): void;
  unMute(): void;
  getCurrentTime(): number;
  getDuration(): number;
  getPlayerState(): number;
  getVideoUrl(): string;
  destroy(): void;
}
interface PlayerOptions {
  host?: string;
  videoId?: string;
  width?: string;
  height?: string;
  playerVars?: Record<string, string | number>;
  events: {
    onReady: (event: { target: YouTubePlayer }) => void;
    onStateChange: (event: { data: number }) => void;
    onError: (event: { data: number }) => void;
    onAutoplayBlocked: () => void;
  };
}
interface YouTubeAPI { Player: new (element: HTMLElement, options: PlayerOptions) => YouTubePlayer; }
declare global { interface Window { YT?: YouTubeAPI; onYouTubeIframeAPIReady?: () => void; } }
let pending: Promise<YouTubeAPI> | undefined;

export function loadYouTube(): Promise<YouTubeAPI> {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (pending) return pending;
  pending = new Promise<YouTubeAPI>((resolve, reject) => {
    const previous = window.onYouTubeIframeAPIReady;
    const script = document.createElement('script');
    const finish = (error?: Error) => {
      clearTimeout(timeout);
      window.onYouTubeIframeAPIReady = previous;
      if (error) { script.remove(); reject(error); }
      else if (window.YT) resolve(window.YT);
    };
    const timeout = setTimeout(() => finish(new Error('YouTube did not respond. Check your connection and retry.')), 20000);
    window.onYouTubeIframeAPIReady = () => { try { previous?.(); } finally { finish(); } };
    script.src = 'https://www.youtube.com/iframe_api';
    script.async = true;
    script.onerror = () => finish(new Error('YouTube could not load. Check your connection or content blocker.'));
    document.head.append(script);
  }).catch(error => { pending = undefined; throw error; });
  return pending;
}

export function youtubeError(code: number): string {
  if (code === 101 || code === 150) return 'The video owner does not allow this clip to play in an embedded player.';
  if (code === 100) return 'This YouTube video is unavailable or private.';
  if (code === 153) return 'YouTube could not identify this page. Open the demo in a regular browser and retry.';
  return `YouTube could not play this clip (error ${code}). Retry or open the source below.`;
}
