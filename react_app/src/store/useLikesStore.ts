import { create } from "zustand";
import { likesArrControl } from "../utils/likesStore";

interface iLikes {
    likesArr: string[];
    changeLikesArr: (likesArr: string[]) => void;
}

export const useLikesStore = create<iLikes>((set) => ({
    likesArr: likesArrControl() ?? [],
    changeLikesArr: (dataArr = []) => set(() => ({likesArr: dataArr}) )
}))