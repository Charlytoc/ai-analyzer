import { create } from "zustand";
import { persist } from "zustand/middleware";
import { generateUniqueID } from "../utils/lib";

type StoreState = {
  clientId: string | null;
  createClientId: () => void;
  clear: () => void;
};

export const useStore = create<StoreState>()(
  persist(
    (set, get) => ({
      clientId: null,
      createClientId: () => {
        const { clientId } = get();
        if (clientId) return;
        const id = generateUniqueID();
        set({ clientId: id });
      },
      clear: () => set({ clientId: null }),
    }),
    {
      name: "app-store",
    }
  )
);
