// store/modelStore.js — global state with Zustand
import { create } from "zustand"

export const useModelStore = create((set) => ({
  modelLoaded:       false,
  modelId:           null,
  modelName:         null,
  modelConfig:       null,
  bits:              3,
  mode:              "ip",
  compressionRatio:  1.0,
  loadingModel:      false,
  loadError:         null,
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

  setBits:  (b) => set({ bits: b }),
  setMode:  (m) => set({ mode: m }),
  setLoading: (v) => set({ loadingModel: v }),
  setError: (e) => set({ loadError: e, loadingModel: false }),
  setLiveMetrics: (m) => set({ liveMetrics: m }),
  reset: () => set({ modelLoaded: false, modelId: null, modelName: null }),
}))
