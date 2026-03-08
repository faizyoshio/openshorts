import React, { useState } from 'react';
import { Youtube, Upload, FileVideo, X } from 'lucide-react';

export default function MediaInput({ onProcess, isProcessing }) {
    const [mode, setMode] = useState('url'); // 'url' | 'file'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [durationMode, setDurationMode] = useState('auto'); // 'auto' | 'custom'
    const [clipDuration, setClipDuration] = useState(30);
    const [countMode, setCountMode] = useState('auto'); // 'auto' | 'custom'
    const [clipCount, setClipCount] = useState(6);

    const handleSubmit = (e) => {
        e.preventDefault();
        const safeDuration = Math.max(10, Math.min(60, Number(clipDuration) || 30));
        const safeCount = Math.max(1, Math.min(20, Number(clipCount) || 6));

        const options = {
            durationMode,
            clipDuration: durationMode === 'custom' ? safeDuration : null,
            countMode,
            clipCount: countMode === 'custom' ? safeCount : null,
        };

        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, options });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, options });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="bg-surface border border-white/5 rounded-2xl p-6 animate-[fadeIn_0.6s_ease-out]">
            <div className="flex gap-4 mb-6 border-b border-white/5 pb-4">
                <button
                    onClick={() => setMode('url')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'
                        }`}
                >
                    <Youtube size={18} />
                    YouTube URL
                </button>
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'
                        }`}
                >
                    <Upload size={18} />
                    Upload File
                </button>
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4">
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://www.youtube.com/watch?v=..."
                            className="input-field"
                            required
                        />
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-white">
                                <FileVideo className="text-primary" />
                                <span className="font-medium">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 hover:bg-white/10 rounded-full"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                                <p className="text-zinc-400">Click to upload or drag and drop</p>
                                <p className="text-xs text-zinc-600 mt-1">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-3">
                        <p className="text-xs uppercase tracking-wide text-zinc-500">Clip Duration</p>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                onClick={() => setDurationMode('auto')}
                                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${durationMode === 'auto' ? 'bg-primary/20 text-primary border border-primary/30' : 'bg-white/5 text-zinc-400 hover:text-white border border-white/10'}`}
                            >
                                Auto
                            </button>
                            <button
                                type="button"
                                onClick={() => setDurationMode('custom')}
                                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${durationMode === 'custom' ? 'bg-primary/20 text-primary border border-primary/30' : 'bg-white/5 text-zinc-400 hover:text-white border border-white/10'}`}
                            >
                                Custom
                            </button>
                        </div>
                        {durationMode === 'custom' && (
                            <div className="space-y-1">
                                <label className="text-[11px] text-zinc-500">Seconds per clip (10 - 60)</label>
                                <input
                                    type="number"
                                    min="10"
                                    max="60"
                                    step="1"
                                    value={clipDuration}
                                    onChange={(e) => setClipDuration(e.target.value)}
                                    className="input-field text-sm py-2"
                                />
                            </div>
                        )}
                        {durationMode === 'auto' && (
                            <p className="text-[11px] text-zinc-500">Auto calculates duration from total source length.</p>
                        )}
                    </div>

                    <div className="bg-black/20 border border-white/5 rounded-xl p-4 space-y-3">
                        <p className="text-xs uppercase tracking-wide text-zinc-500">Output Count</p>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                onClick={() => setCountMode('auto')}
                                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${countMode === 'auto' ? 'bg-primary/20 text-primary border border-primary/30' : 'bg-white/5 text-zinc-400 hover:text-white border border-white/10'}`}
                            >
                                Auto
                            </button>
                            <button
                                type="button"
                                onClick={() => setCountMode('custom')}
                                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${countMode === 'custom' ? 'bg-primary/20 text-primary border border-primary/30' : 'bg-white/5 text-zinc-400 hover:text-white border border-white/10'}`}
                            >
                                Custom
                            </button>
                        </div>
                        {countMode === 'custom' && (
                            <div className="space-y-1">
                                <label className="text-[11px] text-zinc-500">Number of clips (1 - 20)</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="20"
                                    step="1"
                                    value={clipCount}
                                    onChange={(e) => setClipCount(e.target.value)}
                                    className="input-field text-sm py-2"
                                />
                            </div>
                        )}
                        {countMode === 'auto' && (
                            <p className="text-[11px] text-zinc-500">Auto calculates output count from total source length.</p>
                        )}
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={isProcessing || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-6 flex items-center justify-center gap-2"
                >
                    {isProcessing ? (
                        <>
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
