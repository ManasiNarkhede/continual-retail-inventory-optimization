import { expect, test } from '@playwright/test'

test('forecast to simulation flow works', async ({ page }) => {
  test.setTimeout(180_000)

  await page.goto('/')

  const forecastLink = page.getByRole('link', { name: 'Forecast' })
  await expect(forecastLink).toBeVisible({ timeout: 120_000 })
  await forecastLink.click()
  await expect(page.getByRole('heading', { name: 'Demand Forecast Workspace' })).toBeVisible()

  await page.getByRole('button', { name: /Store / }).first().click()

  await page.getByRole('link', { name: 'Simulation' }).click()
  await expect(page.getByRole('heading', { name: 'RL Simulation' })).toBeVisible()

  await page.getByRole('button', { name: 'Run Simulation' }).click()
  await expect(page.getByText('Simulation Summary')).toBeVisible({ timeout: 30_000 })
})
