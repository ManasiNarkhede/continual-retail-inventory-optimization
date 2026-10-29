import { chromium } from '@playwright/test'
import fs from 'fs'
import path from 'path'

const resultsDir = path.join(process.cwd(), 'e2e-results')
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true })

const verdict = {
  flows: [],
  screenshots: [],
  errors: []
}

async function capture(page, name) {
  const file = path.join(resultsDir, `${name}.png`)
  await page.screenshot({ path: file, fullPage: true })
  verdict.screenshots.push(name)
  console.log(`Screenshot: ${file}`)
}

async function checkConsoleErrors(page, flowName) {
  const errors = []
  page.on('pageerror', err => errors.push(`PAGE ERROR: ${err.message}`))
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(`CONSOLE ERROR: ${msg.text()}`)
  })
  // Wait a bit to collect errors
  await page.waitForTimeout(1000)
  if (errors.length > 0) {
    verdict.errors.push({ flow: flowName, errors })
  }
  return errors.length === 0
}

(async () => {
  const browser = await chromium.launch()
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } })
  const page = await context.newPage()

  // Flow 1: Dashboard Load
  try {
    await page.goto('http://127.0.0.1:5173/')
    await page.waitForLoadState('networkidle')
    const heading = await page.locator('h1, [role="main"]').first().isVisible().catch(() => false)
    const noErrors = await checkConsoleErrors(page, 'dashboard-load')
    await capture(page, '01-dashboard-load')
    verdict.flows.push({ name: 'Dashboard Load', pass: heading && noErrors })
  } catch (e) {
    verdict.flows.push({ name: 'Dashboard Load', pass: false, error: e.message })
  }

  // Flow 2: Navigation - All Pages
  const pages = [
    { path: '/forecast', label: 'Forecast', heading: 'Demand Forecast Workspace' },
    { path: '/simulation', label: 'Simulation', heading: 'Inventory Simulation' },
    { path: '/cl-metrics', label: 'CL Metrics', heading: 'Continual Learning Metrics' },
    { path: '/hpo', label: 'HPO', heading: 'Hyperparameter Optimization' },
  ]

  for (const p of pages) {
    try {
      await page.goto(`http://127.0.0.1:5173${p.path}`)
      await page.waitForLoadState('networkidle')
      const headingVisible = await page.getByRole('heading', { name: p.heading }).isVisible().catch(() => false)
      const noErrors = await checkConsoleErrors(page, `${p.label.toLowerCase()}-load`)
      await capture(page, `02-${p.label.toLowerCase()}-load`)
      verdict.flows.push({ name: `${p.label} Load`, pass: headingVisible && noErrors })
    } catch (e) {
      verdict.flows.push({ name: `${p.label} Load`, pass: false, error: e.message })
    }
  }

  // Flow 3: Navigation via Links
  try {
    await page.goto('http://127.0.0.1:5173/')
    await page.waitForLoadState('networkidle')
    await page.getByRole('link', { name: 'Forecast' }).click()
    await page.waitForURL('**/forecast')
    await page.waitForTimeout(1500) // Wait for lazy load
    const forecastVisible = await page.getByRole('heading', { name: 'Demand Forecast Workspace' }).isVisible().catch(() => false)
    await capture(page, '03-nav-forecast')
    verdict.flows.push({ name: 'Nav Forecast', pass: forecastVisible })

    await page.getByRole('link', { name: 'Simulation' }).click()
    await page.waitForURL('**/simulation')
    await page.waitForTimeout(1500) // Wait for lazy load
    const simVisible = await page.getByRole('heading', { name: 'Inventory Simulation' }).isVisible().catch(() => false)
    await capture(page, '04-nav-simulation')
    verdict.flows.push({ name: 'Nav Simulation', pass: simVisible })
  } catch (e) {
    verdict.flows.push({ name: 'Navigation', pass: false, error: e.message })
  }

  // Flow 4: Dark Mode Toggle
  try {
    await page.goto('http://127.0.0.1:5173/')
    await page.waitForLoadState('networkidle')
    const toggle = page.getByRole('button', { name: 'Toggle dark mode' })
    await toggle.click()
    await page.waitForTimeout(500)
    const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'))
    await capture(page, '05-dark-mode-on')
    verdict.flows.push({ name: 'Dark Mode On', pass: isDark })

    await toggle.click()
    await page.waitForTimeout(500)
    const isLight = !await page.evaluate(() => document.documentElement.classList.contains('dark'))
    await capture(page, '06-dark-mode-off')
    verdict.flows.push({ name: 'Dark Mode Off', pass: isLight })
  } catch (e) {
    verdict.flows.push({ name: 'Dark Mode', pass: false, error: e.message })
  }

  // Flow 5: Mobile Navigation
  try {
    const mobileContext = await browser.newContext({ viewport: { width: 375, height: 667 } })
    const mobilePage = await mobileContext.newPage()
    await mobilePage.goto('http://127.0.0.1:5173/')
    await mobilePage.waitForLoadState('networkidle')
    
    // Mobile nav button should be visible (but it's NOT integrated in App.jsx)
    const menuBtn = mobilePage.locator('button[aria-label="Toggle navigation"]')
    const menuVisible = await menuBtn.isVisible().catch(() => false)
    
    if (menuVisible) {
      await menuBtn.click()
      await mobilePage.waitForTimeout(500)
      await capture(mobilePage, '07-mobile-nav-open')
      
      // Click a link in mobile nav
      await mobilePage.getByRole('link', { name: 'Forecast' }).click()
      await mobilePage.waitForURL('**/forecast')
      await capture(mobilePage, '08-mobile-nav-forecast')
      verdict.flows.push({ name: 'Mobile Nav Open', pass: true })
      verdict.flows.push({ name: 'Mobile Nav Navigate', pass: true })
    } else {
      // MobileNav component exists but is NOT used in App.jsx
      await capture(mobilePage, '07-mobile-nav-missing')
      verdict.flows.push({ name: 'Mobile Nav Open', pass: false, error: 'MobileNav component exists but is not integrated into App.jsx' })
      verdict.flows.push({ name: 'Mobile Nav Navigate', pass: false, error: 'MobileNav component not integrated' })
    }
    
    await mobileContext.close()
  } catch (e) {
    verdict.flows.push({ name: 'Mobile Nav', pass: false, error: e.message })
  }

  // Flow 6: Offline Banner
  try {
    const offlineContext = await browser.newContext({ viewport: { width: 1280, height: 720 } })
    const offlinePage = await offlineContext.newPage()
    
    // Simulate offline before loading
    await offlineContext.setOffline(true)
    
    try {
      await offlinePage.goto('http://127.0.0.1:5173/')
    } catch (e) {
      // Expected error when offline
    }
    
    await offlinePage.waitForTimeout(2000)
    
    // Check for offline banner or any error indicator
    const banner = offlinePage.locator('text=/Connection lost|offline|retry/i')
    const bannerVisible = await banner.first().isVisible().catch(() => false)
    await capture(offlinePage, '09-offline-banner')
    
    if (!bannerVisible) {
      verdict.flows.push({ name: 'Offline Banner', pass: false, error: 'OfflineBanner component exists but is not integrated into App.jsx' })
    } else {
      verdict.flows.push({ name: 'Offline Banner', pass: true })
    }
    
    await offlineContext.setOffline(false)
    await offlineContext.close()
  } catch (e) {
    verdict.flows.push({ name: 'Offline Banner', pass: false, error: e.message })
  }

  await browser.close()

  // Summary
  const passed = verdict.flows.filter(f => f.pass).length
  const total = verdict.flows.length
  
  console.log('\n========================================')
  console.log('E2E VERIFICATION SUMMARY')
  console.log('========================================')
  verdict.flows.forEach(f => {
    const status = f.pass ? 'PASS' : 'FAIL'
    console.log(`[${status}] ${f.name}`)
    if (f.error) console.log(`       Error: ${f.error}`)
  })
  console.log('----------------------------------------')
  console.log(`Flows: ${passed}/${total}`)
  console.log(`Screenshots: ${verdict.screenshots.length}`)
  console.log(`VERDICT: ${passed === total ? 'PASS' : 'PARTIAL'}`)
  
  if (verdict.errors.length > 0) {
    console.log('\nConsole Errors Detected:')
    verdict.errors.forEach(e => {
      console.log(`  ${e.flow}:`)
      e.errors.forEach(err => console.log(`    - ${err}`))
    })
  }
  
  fs.writeFileSync(path.join(resultsDir, 'verdict.json'), JSON.stringify(verdict, null, 2))
})()
