import {readFileSync} from 'fs';
import {minimatch} from 'minimatch';

describe('firebase.json', () => {
  const filePath = '../firebase.json';
  let fileContent: string;

  beforeEach(() => {
    fileContent = readFileSync(filePath, 'utf8');
  });

  it('firebase.json should be a valid JSON file', async () => {
    expect(JSON.parse(fileContent)).toBeTruthy();
  });

  it('firebase.json should have rewrites', async () => {
    const obj = JSON.parse(fileContent);
    expect(obj.hosting.rewrites).toBeTruthy();
  });

  function resolveRewrite(path: string) {
    const rewrites = JSON.parse(fileContent).hosting.rewrites;

    for (const rewrite of rewrites) {
      if (minimatch(path, rewrite.source)) {
        return rewrite.function ?? rewrite.destination;
      }
    }
    return path;
  }

  it('firebase.json should redirect "/random-path" to the index.html', async () => {
    expect(resolveRewrite('/random-path')).toEqual('/index.html');
  });

  it('firebase.json should redirect "/api/spoken-to-signed" to the translate-textToText', async () => {
    expect(resolveRewrite('/api/spoken-to-signed')).toEqual('translate-textToText');
  });

  it('firebase.json should not redirect assets to the index.html', async () => {
   
    expect(resolveRewrite('/assets/random-path')).toEqual('/assets/random-path');
  });

  it('firebase.json should not redirect api to the index.html', async () => {
    expect(resolveRewrite('/api/some-service')).toEqual('/api/some-service');
  });
});
