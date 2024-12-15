module Main exposing (..)

import Browser
import Html
import Html.Attributes
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Parser exposing ((|.), (|=))
import Random
import Url.Builder


imageOrigin =
    "http://localhost:8501"


numberOfExamples =
    20210


pixelSize =
    28


randomExample : Random.Generator Int
randomExample =
    Random.int 0 (numberOfExamples - 1)


type Msg
    = CellHover Int
    | ResetHoveredCell
    | CellClick Int
    | GotSaeExamplesEventId (Result Http.Error String)
    | GotRawSaeExamples (Result Http.Error String)
    | ParsedSaeExamples (Result D.Error (List String))
    | GetRandomExample
    | GotRandomExample Int


type alias Model =
    { err : Maybe String
    , example : Int
    , saeUrls : List String
    , hoveredCell : Maybe Int
    , selectedCell : Maybe Int
    , sliderValues : List Float
    }


main =
    Browser.document
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        }


init : () -> ( Model, Cmd Msg )
init _ =
    ( { err = Nothing
      , example = 1
      , saeUrls = []
      , hoveredCell = Nothing
      , selectedCell = Nothing
      , sliderValues = []
      }
    , Cmd.none
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        CellHover i ->
            ( { model | hoveredCell = Just i }, Cmd.none )

        ResetHoveredCell ->
            ( { model | hoveredCell = Nothing }, Cmd.none )

        CellClick i ->
            ( { model | selectedCell = Just i }
            , getSaeExamples (model.example - 1) i
            )

        GotSaeExamplesEventId result ->
            case result of
                Ok id ->
                    ( model, getSaeExamplesResult id )

                _ ->
                    ( model, Cmd.none )

        GotRawSaeExamples result ->
            case result of
                Ok raw ->
                    case Parser.run eventParser raw of
                        Ok body ->
                            update (ParsedSaeExamples (D.decodeString (D.list (D.field "url" D.string)) (Debug.log "parsed" body))) model

                        Err _ ->
                            ( model, Cmd.none )

                Err err ->
                    let
                        errMsg =
                            httpErrToString err "get SAE examples"
                    in
                    ( { model | err = Just errMsg }, Cmd.none )

        ParsedSaeExamples result ->
            case result of
                Ok urls ->
                    ( { model | saeUrls = urls }, Cmd.none )

                Err err ->
                    ( { model | err = Just (Debug.toString err) }, Cmd.none )

        GotRandomExample i ->
            ( { model | example = i }, Cmd.none )

        GetRandomExample ->
            ( { model | selectedCell = Nothing, saeUrls = [] }, Random.generate GotRandomExample randomExample )


httpErrToString : Http.Error -> String -> String
httpErrToString err description =
    case err of
        Http.BadUrl url ->
            "Could not " ++ description ++ " because URL '" ++ url ++ "' was wrong."

        Http.Timeout ->
            "Could not " ++ description ++ " because request timed out."

        Http.NetworkError ->
            "Could not " ++ description ++ " because of a generic network error."

        Http.BadStatus status ->
            "Could not " ++ description ++ ": Status " ++ String.fromInt status ++ "."

        Http.BadBody explanation ->
            "Could not " ++ description ++ ": " ++ explanation ++ "."


getSaeExamples : Int -> Int -> Cmd Msg
getSaeExamples img cell =
    Http.post
        { url = "http://127.0.0.1:7860/gradio_api/call/get-sae-examples"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "data", E.list E.int [ img, cell ] ) ]
                )
        , expect = Http.expectJson GotSaeExamplesEventId eventIdDecoder
        }


getSaeExamplesResult : String -> Cmd Msg
getSaeExamplesResult id =
    Http.get
        { url = "http://127.0.0.1:7860/gradio_api/call/get-sae-examples/" ++ id
        , expect = Http.expectString GotRawSaeExamples
        }


eventIdDecoder : D.Decoder String
eventIdDecoder =
    D.field "event_id" D.string


saeImagesDecoder : D.Decoder (List String)
saeImagesDecoder =
    D.list (D.field "url" D.string)


view : Model -> Browser.Document Msg
view model =
    { title = "Semantic Segmentation"
    , body =
        [ Html.header [] []
        , Html.main_
            []
            [ viewExample model
            , viewCell model.hoveredCell
            , viewCell model.selectedCell
            , viewErr model.err
            , Html.button
                [ Html.Events.onClick GetRandomExample ]
                [ Html.text "Random Example" ]
            , Html.div [] (List.map viewImage model.saeUrls)
            ]
        ]
    }


viewErr : Maybe String -> Html.Html Msg
viewErr err =
    case err of
        Just msg ->
            Html.p [] [ Html.text msg ]

        Nothing ->
            Html.p [] [ Html.text "No errors." ]


viewCell : Maybe Int -> Html.Html Msg
viewCell cell =
    case cell of
        Just i ->
            Html.p [] [ Html.text (String.fromInt i) ]

        Nothing ->
            Html.p [] [ Html.text "Hover over a cell." ]


viewExample : Model -> Html.Html Msg
viewExample model =
    Html.div
        [ Html.Attributes.id "img-grid-wrapper" ]
        [ Html.div
            [ Html.Attributes.id "img-grid-overlay"
            ]
            (List.map (viewGridCell model.hoveredCell model.selectedCell) (List.range 0 255))
        , Html.img
            [ Html.Attributes.src ("http://localhost:8501/images/training/ADE_train_" ++ (String.fromInt model.example |> String.padLeft 8 '0') ++ ".jpg")
            , Html.Attributes.style "display" "block"
            , Html.Attributes.style "width" "448px"
            , Html.Attributes.style "height" "448px"
            ]
            []
        ]


viewGridCell : Maybe Int -> Maybe Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        tint =
            case ( hovered, selected ) of
                ( Just h, Nothing ) ->
                    if h == self then
                        "rgb(255 0 0 / 0.5)"

                    else
                        "rgb(0 0 0 / 0)"

                ( Nothing, Just s ) ->
                    if s == self then
                        "rgb(0 255 0 / 0.5)"

                    else
                        "rgb(0 0 0 / 0)"

                ( Just h, Just s ) ->
                    if h == self && s == self then
                        "rgb(255 255 0 / 0.5)"

                    else if h == self && s /= self then
                        "rgb(255 0 0 / 0.5)"

                    else if h /= self && s == self then
                        "rgb(0 255 0 / 0.5)"

                    else
                        "rgb(0 0 0 / 0)"

                _ ->
                    "rgb(0 0 0 / 0)"
    in
    Html.div
        [ Html.Attributes.class "img-grid-cell"
        , Html.Events.onMouseEnter (CellHover self)
        , Html.Events.onMouseLeave ResetHoveredCell
        , Html.Events.onClick (CellClick self)
        , Html.Attributes.style "background-color" tint
        ]
        []


viewImage : String -> Html.Html Msg
viewImage url =
    Html.img
        [ Html.Attributes.src (String.replace "gradio_api/gradio_api" "gradio_api" url) ]
        []


eventParser : Parser.Parser String
eventParser =
    Parser.succeed identity
        |. Parser.keyword "event"
        |. Parser.symbol ":"
        |. Parser.spaces
        |. Parser.keyword "complete"
        |. Parser.spaces
        |. Parser.keyword "data"
        |. Parser.symbol ":"
        |. Parser.spaces
        |= restParser
        |. Parser.spaces
        |. Parser.end


restParser : Parser.Parser String
restParser =
    Parser.getChompedString <|
        Parser.succeed ()
            |. Parser.chompUntilEndOr "\n"
