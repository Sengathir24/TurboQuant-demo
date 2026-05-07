// store/modelStore.js — global state with Zustand
import { create } from "zustand"

export const useModelStore = create((set) => ({
  modelLoaded:       false,
  modelId:           null,
  modelName:         null,
  modelConfig:       null,
  bits:              4,
  mode:              "ip",
  compressionRatio:  1.0,
  loadingModel:      false,
  loadError:         null,
  // liveMetrics includes: avg_mse_k, avg_mse_v, avg_compress_ms,
  //   mb_saved, compression_ratio, n_layers, vram_gb, per_layer[]
  liveMetrics:       {},

  setModelLoaded: (cfg) => set({
    modelLoaded:      true,
    modelId:          cfg.model_id,
    modelName:        cfg.model_name,
    modelConfig:      cfg,
    compressionRatio: cfg.compression_ratio,
    loadingModel:     false,
    loadError:        null,
  }),

  setBits:        (b) => set({ bits: b }),
  setMode:        (m) => set({ mode: m }),
  setLoading:     (v) => set({ loadingModel: v }),
  setError:       (e) => set({ loadError: e, loadingModel: false }),
  // Merge to avoid dropping existing fields between websocket ticks
  setLiveMetrics: (m) => set((state) => ({ liveMetrics: { ...state.liveMetrics, ...m } })),
  reset: () => set({ modelLoaded: false, modelId: null, modelName: null, liveMetrics: {} }),
}))
