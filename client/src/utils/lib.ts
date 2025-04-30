export const generateUniqueID = (): string => {
  const timestamp = Date.now();
  const randomStr = Math.random().toString(36).substring(2, 7);
  return `${timestamp}_${randomStr}`;
};
