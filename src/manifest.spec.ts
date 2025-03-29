describe('manifest.webmanifest', () => {
  const filePath = 'src/manifest.webmanifest';

  it('webmanifest should be a valid JSON file', async () => {
    const res = await fetch('base/' + filePath);
    const txt = await res.text();
    expect(JSON.parse(txt)).toBeTruthy();
  });
});
