# Docs assets

Drop these files here and they'll show up in the main README:

- `hero.png` — hero render (suggested 1560×880 or 1920×1080 still). Take a screenshot of the sphere drop scene at the moment of impact, or a settled pool with sun reflections, or a dam-break mid-flow.
- `demo.gif` — 30-60 second loop showing scene switching, mouse interaction, and the splash sequence. Generate from a screen recording with e.g.:
    ```sh
    ffmpeg -i recording.mov -vf "fps=24,scale=780:-1:flags=lanczos" -loop 0 demo.gif
    ```
