import { test, expect } from 'playwright/test';

test.use({ headless: true });

test.describe('[Wiki][Desktop][MainPage] Random article via main menu navigation', () => {
  test('Opening the Main menu and clicking Random article navigates to a random Wikipedia article', async ({ page }) => {
    const mainMenuButton = page.getByRole('button', { name: 'Main menu' });
    const randomArticleLink = page.getByRole('link', { name: 'Random article' });

    await test.step('Open https://en.wikipedia.org/wiki/Main_Page', async () => {
      await page.goto('https://en.wikipedia.org/wiki/Main_Page');

      await expect(page, 'website loaded').toHaveURL('https://en.wikipedia.org/wiki/Main_Page');
    });

    await test.step('Press menu button', async () => {
      await mainMenuButton.click();

      await expect(randomArticleLink, 'main menu opened').toBeVisible();
    });

    await test.step('Press Random article', async () => {
      await randomArticleLink.click();

      await expect(page, 'random article loaded (url has changed)').not.toHaveURL('https://en.wikipedia.org/wiki/Main_Page');
    });
  });
});
