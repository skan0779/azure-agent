const optimisticThreadTitles = new Map<string, string>();
const listeners = new Set<() => void>();

const emit = () => {
  for (const listener of listeners) {
    listener();
  }
};

export const getOptimisticThreadTitle = (threadId?: string | null) => {
  if (!threadId) {
    return undefined;
  }

  return optimisticThreadTitles.get(threadId);
};

export const setOptimisticThreadTitle = (threadId: string, title: string) => {
  optimisticThreadTitles.set(threadId, title);
  emit();
};

export const clearOptimisticThreadTitle = (threadId?: string | null) => {
  if (!threadId) {
    return;
  }

  if (!optimisticThreadTitles.delete(threadId)) {
    return;
  }

  emit();
};

export const subscribeOptimisticThreadTitles = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
