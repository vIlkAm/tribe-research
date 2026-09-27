import { useCallback, useEffect, useRef, useState } from 'react';
import { clipEndMs, clipFileError, clipSeekMs } from './clip-playback';

interface Options {
  durationMs: number; playing: boolean; speed: number; loop: boolean;
  onTime: (ms: number) => void; onPlaying: (playing: boolean) => void;
}

export function useClipPlayer(options: Options) {
  const video = useRef<HTMLVideoElement>(null);
  const latest = useRef(options);
  latest.current = options;
  const [source, setSource] = useState<{ url: string; name: string } | null>(null);
  const [duration, setDuration] = useState<number>();
  const [ready, setReady] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const active = !!source && !error;
  const endMs = clipEndMs(options.durationMs, active ? duration : undefined);
  const desiredTime = useRef(0);

  useEffect(() => () => { if (source) URL.revokeObjectURL(source.url); }, [source]);

  const choose = useCallback((file: File) => {
    const invalid = clipFileError(file);
    if (invalid) { setNotice(invalid); return; }
    video.current?.pause();
    latest.current.onPlaying(false);
    latest.current.onTime(0);
    desiredTime.current = 0;
    setReady(false); setDuration(undefined); setError(null); setNotice(null); setWaiting(false);
    setSource({ url: URL.createObjectURL(file), name: file.name });
  }, []);

  const remove = useCallback(() => {
    video.current?.pause();
    latest.current.onPlaying(false);
    setSource(null); setReady(false); setDuration(undefined); setError(null); setNotice(null); setWaiting(false);
  }, []);

  const seek = useCallback((ms: number) => {
    const target = clipSeekMs(ms, endMs);
    desiredTime.current = target;
    if (video.current && ready) video.current.currentTime = target / 1000;
    return target;
  }, [endMs, ready]);

  function loaded() {
    const media = video.current;
    if (!media || !Number.isFinite(media.duration) || media.duration <= 0) {
      failed(); return;
    }
    setDuration(media.duration);
    media.currentTime = clipSeekMs(desiredTime.current, clipEndMs(latest.current.durationMs, media.duration)) / 1000;
    setReady(true);
  }

  function failed() {
    latest.current.onPlaying(false);
    setReady(false); setWaiting(false);
    setError('This browser could not play the clip. Try an MP4 with H.264 video or a WebM file.');
  }

  // The video is the clock while attached, including when it buffers or seeks.
  useEffect(() => {
    const media = video.current;
    if (!media || !ready || !active) return;
    let cancelled = false;
    media.playbackRate = options.speed;
    media.muted = muted;
    if (options.playing) {
      setNotice(null);
      void media.play().catch(() => {
        if (cancelled) return;
        latest.current.onPlaying(false);
        setNotice('Playback did not start. Press Play to try again.');
      });
    } else {
      media.pause();
      latest.current.onTime(clipSeekMs(media.currentTime * 1000, endMs));
    }
    return () => { cancelled = true; };
  }, [options.playing, options.speed, ready, active, muted, source, endMs]);

  const finish = useCallback(() => {
    const media = video.current;
    if (!media) return;
    if (latest.current.loop) {
      media.currentTime = 0;
      latest.current.onTime(0);
      if (latest.current.playing) void media.play().catch(() => latest.current.onPlaying(false));
    } else {
      media.pause();
      latest.current.onTime(clipSeekMs(endMs, endMs));
      latest.current.onPlaying(false);
    }
  }, [endMs]);

  useEffect(() => {
    if (!active || !ready || !options.playing) return;
    let frame = 0, last = 0;
    const tick = (now: number) => {
      const media = video.current;
      if (media && !media.seeking && now - last > 30) {
        last = now;
        if (media.currentTime * 1000 >= endMs) {
          finish();
          if (!latest.current.loop) return;
        } else latest.current.onTime(clipSeekMs(media.currentTime * 1000, endMs));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [active, ready, options.playing, endMs, finish]);

  return {
    video, source, active, ready, waiting, muted, error, notice, duration, endMs,
    choose, remove, seek, loaded, failed, finish,
    toggleMute: () => setMuted(value => !value),
    onWaiting: () => setWaiting(true),
    onPlaying: () => setWaiting(false),
    onSeeked: () => {
      if (video.current) latest.current.onTime(clipSeekMs(video.current.currentTime * 1000, endMs));
    },
  };
}

export type ClipPlayer = ReturnType<typeof useClipPlayer>;
