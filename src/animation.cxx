#include <list>
class ParticleEffect
{
	struct Particle
	{
		unsigned int startTime, duration;
		int x, y, startRadius, endRadius;
		uint8_t startOpacity, endOpacity;
		int movementSpeed, movementAngle_degree;
		std::vector<SDL_Texture *> frames;
		bool randomShuffle;
		int lastRenderedFrameIdx;
	};
	std::list<Particle> particles;
	inline void update(Particle &particle);

  public:
	inline void addParticle(unsigned int startTime, unsigned int duration, int x, int y, int startRadius, int endRadius,
							uint8_t startOpacity, uint8_t endOpacity, int movementSpeed, int movementAngle_degree,
							std::vector<SDL_Texture *> frames, bool randomShuffle = 0, int lastRenderedFrameIdx = 0);
	inline void clear();
	void render(SDL_Renderer *s, unsigned int currentTime);
};

inline void ParticleEffect::addParticle(unsigned int startTime, unsigned int duration, int x, int y, int startRadius, int endRadius,
										uint8_t startOpacity, uint8_t endOpacity, int movementSpeed, int movementAngle_degree,
										std::vector<SDL_Texture *> frames, bool randomShuffle, int lastRenderedFrameIdx)
{
	particles.emplace_front(Particle{startTime, duration, x, y, startRadius, endRadius,
									 startOpacity, endOpacity, movementSpeed, movementAngle_degree,
									 frames, randomShuffle, lastRenderedFrameIdx});
}
inline void ParticleEffect::update(Particle &particle)
{
	particle.lastRenderedFrameIdx = (rand() % particle.frames.size()) * (particle.randomShuffle) + (particle.lastRenderedFrameIdx + 1) * ((not particle.randomShuffle) && (particle.lastRenderedFrameIdx < particle.frames.size() - 1));

	particle.x += particle.movementSpeed * cos(particle.movementAngle_degree * M_PI / 180);
	particle.y += particle.movementSpeed * sin(particle.movementAngle_degree * M_PI / 180);
}
inline void ParticleEffect::clear()
{
	particles.clear();
}
void ParticleEffect::render(SDL_Renderer *s, unsigned int currentTime)
{
	auto particle = particles.begin();
	while (particle != particles.end())
	{
		if (currentTime - particle->startTime > particle->duration)
		{
			particle = particles.erase(particle);
			continue;
		}

		SDL_SetTextureAlphaMod(particle->frames[particle->lastRenderedFrameIdx], particle->startOpacity + (particle->endOpacity - particle->startOpacity) * double(currentTime - particle->startTime) / particle->duration);

		int radius = particle->startRadius + (particle->endRadius - particle->startRadius) * double(currentTime - particle->startTime) / particle->duration;
		SDL_Rect temp_rect = createRect(particle->x, particle->y, 2 * radius, 2 * radius);
		SDL_RenderCopy(s, particle->frames[particle->lastRenderedFrameIdx], NULL, &temp_rect);

		update(*particle);
		particle++;
	}
}

#include "shooter.cxx"
void animateDeath(Shooter &player, SDL_Renderer *s, std::unordered_map<std::string, SDL_Texture *> deathImages)
{
	SDL_Rect temp_shooterRect = createRect(player.x, player.y, 2 * Shooter::RADIUS, 2 * Shooter::RADIUS);
	SDL_Rect temp_rect = createRect(player.x, player.y, 2 * Shooter::GUN_RADIUS, 2 * Shooter::GUN_RADIUS);
	std::string frameName = "gunDeath_0";
	for (int frame_no = 1; frame_no <= deathImages.size(); frame_no++)
	{
		SDL_RenderCopyEx(s, player.images["shooter"], NULL, &temp_shooterRect, player.headingAngle_degree, NULL, SDL_FLIP_NONE);
		frameName[9] = frame_no + int('0');
		SDL_RenderCopyEx(s, deathImages[frameName], NULL, &temp_rect, player.gunAngle_degree, NULL, SDL_FLIP_NONE);
		SDL_RenderPresent(s);
		SDL_RenderClear(s);
		SDL_Delay(500);
	}
	player.images["gun"] = deathImages[frameName];
}