param (
    [string]$repoName = "pc-control-system",
    [string]$description = "Система управления ПК на основе распознавания элементов интерфейса",
    [string]$githubUsername,
    [string]$githubToken
)

# Проверка наличия имени пользователя и токена
if (-not $githubUsername -or -not $githubToken) {
    Write-Host "Необходимо указать имя пользователя GitHub и токен доступа."
    Write-Host "Пример: .\create_github_repo.ps1 -githubUsername YOUR_USERNAME -githubToken YOUR_TOKEN"
    exit 1
}

# Формирование данных для создания репозитория
$body = @{
    name        = $repoName
    description = $description
    private     = $false
    auto_init   = $false
} | ConvertTo-Json

Write-Host "Создание репозитория $repoName..."

# Создание репозитория через API GitHub
try {
    $secureToken = ConvertTo-SecureString -String $githubToken -AsPlainText -Force
    $cred = New-Object System.Management.Automation.PSCredential($githubUsername, $secureToken)

    $params = @{
        Uri            = "https://api.github.com/user/repos"
        Method         = "POST"
        ContentType    = "application/json"
        Body           = $body
        Authentication = "Basic"
        Credential     = $cred
        Headers        = @{
            "Accept" = "application/vnd.github.v3+json"
        }
    }

    $response = Invoke-RestMethod @params
    
    Write-Host "Репозиторий успешно создан: $($response.html_url)"
    
    # Настройка удаленного репозитория Git
    Write-Host "Настройка удаленного репозитория..."
    git remote add origin "https://github.com/$githubUsername/$repoName.git"
    
    # Переименование ветки в main, если это не было сделано
    git branch -M main
    
    # Настройка push с использованием токена для аутентификации
    $remoteUrl = "https://$($githubUsername):$($githubToken)@github.com/$($githubUsername)/$($repoName).git"
    
    # Обновление URL для использования токена
    git remote set-url origin $remoteUrl
    
    # Отправка кода в репозиторий
    Write-Host "Отправка кода в репозиторий..."
    git push -u origin main
    
    Write-Host "Процесс завершен! Ваш репозиторий доступен по адресу: $($response.html_url)"
} 
catch {
    Write-Host "Произошла ошибка при создании репозитория:"
    Write-Host $_
    exit 1
} 