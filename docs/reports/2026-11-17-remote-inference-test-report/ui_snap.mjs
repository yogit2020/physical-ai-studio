import { chromium } from '/home/user1/yogini/pas-safety/application/ui/node_modules/playwright/index.mjs';
const OUT = '/tmp/uisnap';
const URL = 'http://localhost:7860/projects/b40d289d-536f-4411-88dc-8c4af490b597/models';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 950 }, deviceScaleFactor: 1 });
await page.goto(URL, { waitUntil: 'networkidle' });
const row = page.locator("xpath=//*[normalize-space(text())='Pi0.5 210base 65ft']/ancestor::*[.//button[normalize-space()='Run model']][1]");
await row.getByRole('button', { name: 'Run model' }).first().click();
const dialog = page.getByRole('dialog');
await dialog.waitFor();
await page.waitForTimeout(800);
await dialog.screenshot({ path: `${OUT}/1-dialog-default.png` });

for (const [label, server] of [['PyTorch', '18767'], ['OpenVINO', '18765']]) {
  await dialog.getByText(label, { exact: true }).click();
  await page.waitForTimeout(400);
  await dialog.getByRole('button', { name: /Inference device/ }).click();
  const option = page.getByRole('option', { name: new RegExp(`Remote server .*${server}`) });
  await option.waitFor();
  if (label === 'PyTorch') { await page.waitForTimeout(700); await page.screenshot({ path: `${OUT}/2-device-list-torch.png`, clip: { x: 430, y: 230, width: 640, height: 500 } }); }
  await option.click();
  await page.waitForTimeout(800);
  await dialog.getByRole('button', { name: /Test connection/ }).click();
  await dialog.getByText(/Connected|Not reachable/).waitFor({ timeout: 20000 });
  await page.waitForTimeout(300);
  await dialog.screenshot({ path: `${OUT}/3-${label.toLowerCase()}-remote-tested.png` });
  console.log(label, 'result:', await dialog.getByText(/Connected|Not reachable/).innerText());
}
await browser.close();
