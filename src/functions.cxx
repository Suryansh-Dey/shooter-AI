inline int approximateInt(float val)
{
	return (val + 0.5) * (val >= 0) + (val - 0.5) * (val < 0);
}
inline int angleOfVector(float x, float y)
{
	return atan(y / x) * 180 / M_PI - 180 * (x < 0);
}
void FPS_manager(const int FRAME_GAP)
{
	static int last_time = 0;
	int delay = SDL_GetTicks() - last_time;
	if (delay < FRAME_GAP)
	{
		SDL_Delay(FRAME_GAP - delay);
	}
	last_time = SDL_GetTicks();
}
inline SDL_Rect createRect(int x, int y, int w, int h)
{
	SDL_Rect rect{x - w / 2, y - h / 2, w, h};
	return rect;
}
