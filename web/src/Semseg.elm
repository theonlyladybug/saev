module Main exposing (..)

import Array
import Browser
import Dict
import Html
import Html.Attributes
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Parser exposing ((|.), (|=))
import Random
import Set


main =
    Browser.document
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        }



-- MESSAGE


type Msg
    = HoverPatch Int
    | ResetHoveredPatch
    | ToggleSelectedPatch Int
    | ResetSelectedPatches
    | GotEventId (String -> Cmd Msg) (Result Http.Error String)
    | GotEventData (String -> Msg) (Result Http.Error String)
    | ParsedSaeExamples (Result D.Error (List SaeExampleResult))
    | ParsedImageUrl (Result D.Error (Maybe String))
    | ParsedTrueLabels (Result D.Error (List SegmentationResult))
    | ParsedPredLabels (Result D.Error (List SegmentationResult))
    | ParsedModifiedLabels (Result D.Error (List SegmentationResult))
    | GetRandomExample
    | SetExample Int
    | SetExampleInput String
    | SetSlider Int String



-- MODEL


type alias Model =
    { err : Maybe String
    , exampleIndex : Int
    , hoveredPatchIndex : Maybe Int
    , selectedPatchIndices : Set.Set Int
    , saeLatents : List SaeLatent
    , imageUrl : Maybe String

    -- Semantic segmenations
    , trueLabelsUrl : Maybe String
    , predLabelsUrl : Maybe String
    , predLabels : List Int
    , modifiedLabelsUrl : Maybe String
    , modifiedLabels : List Int

    -- Progression
    , nPatchesExplored : Int
    , nPatchResets : Int
    , nImagesExplored : Int

    -- UI
    , sliders : Dict.Dict Int Float
    }


type alias SaeLatent =
    { latent : Int
    , urls : List String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    let
        example =
            3122
    in
    ( { err = Nothing
      , exampleIndex = example
      , hoveredPatchIndex = Nothing
      , selectedPatchIndices = Set.empty
      , saeLatents = []
      , imageUrl = Nothing
      , trueLabelsUrl = Nothing
      , predLabelsUrl = Nothing
      , predLabels = []
      , modifiedLabelsUrl = Nothing
      , modifiedLabels = []

      -- Progression
      , nPatchesExplored = 0
      , nPatchResets = 0
      , nImagesExplored = 0

      -- UI
      , sliders = Dict.empty
      }
    , Cmd.batch
        [ getImageUrl example, getPredLabels example ]
    )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        HoverPatch i ->
            ( { model | hoveredPatchIndex = Just i }, Cmd.none )

        ResetHoveredPatch ->
            ( { model | hoveredPatchIndex = Nothing }, Cmd.none )

        ToggleSelectedPatch i ->
            let
                ( patchIndices, nPatchesExplored, nPatchResets ) =
                    if Set.member i model.selectedPatchIndices then
                        ( Set.remove i model.selectedPatchIndices
                        , model.nPatchesExplored
                        , if Set.size model.selectedPatchIndices == 1 then
                            model.nPatchResets + 1

                          else
                            model.nPatchResets
                        )

                    else
                        ( Set.insert i model.selectedPatchIndices
                        , model.nPatchesExplored + 1
                        , model.nPatchResets
                        )
            in
            ( { model
                | selectedPatchIndices = patchIndices
                , saeLatents = []
                , modifiedLabelsUrl = Nothing
                , nPatchesExplored = nPatchesExplored
                , nPatchResets = nPatchResets
              }
            , getSaeExamples model.exampleIndex patchIndices
            )

        ResetSelectedPatches ->
            ( { model
                | selectedPatchIndices = Set.empty
                , saeLatents = []
                , modifiedLabelsUrl = Nothing
                , nPatchResets = model.nPatchResets + 1
              }
            , Cmd.none
            )

        GotEventId fn result ->
            case result of
                Ok id ->
                    ( model, fn id )

                Err err ->
                    let
                        errMsg =
                            httpErrToString err "get event id"
                    in
                    ( { model | err = Just errMsg }, Cmd.none )

        GotEventData decoderFn result ->
            case result of
                Ok raw ->
                    case Parser.run eventDataParser raw of
                        Ok json ->
                            update (decoderFn json) model

                        Err _ ->
                            ( model, Cmd.none )

                Err err ->
                    let
                        errMsg =
                            httpErrToString err "get event result"
                    in
                    ( { model | err = Just errMsg }, Cmd.none )

        ParsedSaeExamples result ->
            case result of
                Ok results ->
                    let
                        latents =
                            parseSaeExampleResults results

                        sliders =
                            latents
                                |> List.map (\latent -> ( latent.latent, 0.0 ))
                                |> Dict.fromList
                    in
                    ( { model | saeLatents = latents, sliders = sliders }, Cmd.none )

                Err err ->
                    ( { model | err = Just (D.errorToString err) }, Cmd.none )

        ParsedTrueLabels result ->
            case result of
                Ok results ->
                    let
                        url =
                            results |> List.filterMap segmentationResultToUrl |> List.head
                    in
                    ( { model | trueLabelsUrl = url }, Cmd.none )

                Err err ->
                    ( { model | trueLabelsUrl = Nothing, err = Just (D.errorToString err) }
                    , Cmd.none
                    )

        ParsedPredLabels result ->
            case result of
                Ok results ->
                    let
                        url =
                            results
                                |> List.filterMap segmentationResultToUrl
                                |> List.head

                        labels =
                            results
                                |> List.filterMap segmentationResultToLabels
                                |> List.head
                                |> Maybe.withDefault []
                    in
                    ( { model | predLabelsUrl = url, predLabels = labels }, Cmd.none )

                Err err ->
                    ( { model | predLabelsUrl = Nothing, err = Just (D.errorToString err) }
                    , Cmd.none
                    )

        ParsedModifiedLabels result ->
            case result of
                Ok results ->
                    let
                        url =
                            results
                                |> List.filterMap segmentationResultToUrl
                                |> List.head

                        labels =
                            results
                                |> List.filterMap segmentationResultToLabels
                                |> List.head
                                |> Maybe.withDefault []
                    in
                    ( { model | modifiedLabelsUrl = url, modifiedLabels = labels }, Cmd.none )

                Err err ->
                    ( { model | modifiedLabelsUrl = Nothing, err = Just (D.errorToString err) }, Cmd.none )

        ParsedImageUrl result ->
            case result of
                Ok maybeUrl ->
                    ( { model | imageUrl = maybeUrl }, Cmd.none )

                Err err ->
                    ( { model | imageUrl = Nothing, err = Just (D.errorToString err) }, Cmd.none )

        SetExample i ->
            ( { model | exampleIndex = i }
            , Cmd.batch
                [ getImageUrl i

                -- , getTrueLabels i
                , getPredLabels i
                ]
            )

        SetExampleInput str ->
            case String.toInt str of
                Just i ->
                    ( { model | exampleIndex = i }
                    , Cmd.batch
                        [ getImageUrl i

                        -- , getTrueLabels i
                        , getPredLabels i
                        ]
                    )

                Nothing ->
                    ( model, Cmd.none )

        GetRandomExample ->
            ( { model
                | imageUrl = Nothing
                , selectedPatchIndices = Set.empty
                , saeLatents = []
              }
            , Random.generate SetExample randomExample
            )

        SetSlider i str ->
            case String.toFloat str of
                Just f ->
                    let
                        sliders =
                            Dict.insert i f model.sliders
                    in
                    ( { model | sliders = sliders }
                    , getModifiedLabels model.exampleIndex sliders
                    )

                Nothing ->
                    ( model, Cmd.none )


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


encodeArgs : List E.Value -> E.Value
encodeArgs args =
    E.object [ ( "data", E.list identity args ) ]


getImageUrl : Int -> Cmd Msg
getImageUrl img =
    Http.post
        { url = "http://127.0.0.1:7860/gradio_api/call/get-image"
        , body =
            Http.jsonBody (encodeArgs [ E.int img ])
        , expect = Http.expectJson (GotEventId getImageUrlResult) eventIdDecoder
        }


getImageUrlResult : String -> Cmd Msg
getImageUrlResult id =
    Http.get
        { url = "http://127.0.0.1:7860/gradio_api/call/get-image/" ++ id
        , expect =
            Http.expectString
                (GotEventData
                    (D.decodeString (D.list imgUrlDecoder)
                        >> Result.map List.head
                        >> ParsedImageUrl
                    )
                )
        }


getTrueLabels : Int -> Cmd Msg
getTrueLabels img =
    Http.post
        { url = "http://127.0.0.1:7860/gradio_api/call/get-true-labels"
        , body =
            Http.jsonBody (encodeArgs [ E.int img ])
        , expect = Http.expectJson (GotEventId getTrueLabelsResult) eventIdDecoder
        }


getTrueLabelsResult : String -> Cmd Msg
getTrueLabelsResult id =
    Http.get
        { url = "http://127.0.0.1:7860/gradio_api/call/get-true-labels/" ++ id
        , expect =
            Http.expectString
                (GotEventData
                    (D.decodeString (D.list segmentationResultDecoder)
                        >> ParsedTrueLabels
                    )
                )
        }


getPredLabels : Int -> Cmd Msg
getPredLabels img =
    Http.post
        { url = "http://127.0.0.1:7860/gradio_api/call/get-pred-labels"
        , body =
            Http.jsonBody (encodeArgs [ E.int img ])
        , expect = Http.expectJson (GotEventId getPredLabelsResult) eventIdDecoder
        }


getPredLabelsResult : String -> Cmd Msg
getPredLabelsResult id =
    Http.get
        { url = "http://127.0.0.1:7860/gradio_api/call/get-pred-labels/" ++ id
        , expect =
            Http.expectString
                (GotEventData
                    (D.decodeString (D.list segmentationResultDecoder)
                        >> ParsedPredLabels
                    )
                )
        }


getModifiedLabels : Int -> Dict.Dict Int Float -> Cmd Msg
getModifiedLabels img sliders =
    let
        ( latents, values ) =
            sliders
                |> Dict.toList
                |> List.unzip
                |> Tuple.mapBoth Array.fromList Array.fromList
    in
    Http.post
        { url = "http://127.0.0.1:7860/gradio_api/call/get-modified-labels"
        , body =
            Http.jsonBody
                (encodeArgs
                    [ E.int img
                    , E.int (Array.get 0 latents |> Maybe.withDefault -1)
                    , E.int (Array.get 1 latents |> Maybe.withDefault -1)
                    , E.int (Array.get 2 latents |> Maybe.withDefault -1)
                    , E.float (Array.get 0 values |> Maybe.withDefault 0.0)
                    , E.float (Array.get 1 values |> Maybe.withDefault 0.0)
                    , E.float (Array.get 2 values |> Maybe.withDefault 0.0)
                    ]
                )
        , expect = Http.expectJson (GotEventId getModifiedLabelsResult) eventIdDecoder
        }


getModifiedLabelsResult : String -> Cmd Msg
getModifiedLabelsResult id =
    Http.get
        { url = "http://127.0.0.1:7860/gradio_api/call/get-modified-labels/" ++ id
        , expect =
            Http.expectString
                (GotEventData
                    (D.decodeString (D.list segmentationResultDecoder)
                        >> ParsedModifiedLabels
                    )
                )
        }


getSaeExamples : Int -> Set.Set Int -> Cmd Msg
getSaeExamples img patches =
    Http.post
        { url = "http://127.0.0.1:7860/gradio_api/call/get-sae-examples"
        , body =
            Http.jsonBody
                (encodeArgs
                    [ E.int img
                    , Set.toList patches |> E.list E.int
                    ]
                )
        , expect = Http.expectJson (GotEventId getSaeExamplesResult) eventIdDecoder
        }


getSaeExamplesResult : String -> Cmd Msg
getSaeExamplesResult id =
    Http.get
        { url = "http://127.0.0.1:7860/gradio_api/call/get-sae-examples/" ++ id
        , expect =
            Http.expectString
                (GotEventData
                    (D.decodeString (D.list saeExampleResultDecoder)
                        >> ParsedSaeExamples
                    )
                )
        }


type SaeExampleResult
    = SaeExampleMissing
    | SaeExampleUrl String
    | SaeExampleLatent Int


saeExampleOutputToUrl : SaeExampleResult -> Maybe String
saeExampleOutputToUrl result =
    case result of
        SaeExampleUrl url ->
            Just url

        _ ->
            Nothing


saeExampleOutputToLatent : SaeExampleResult -> Maybe Int
saeExampleOutputToLatent result =
    case result of
        SaeExampleLatent latent ->
            Just latent

        _ ->
            Nothing


saeExampleResultDecoder : D.Decoder SaeExampleResult
saeExampleResultDecoder =
    D.oneOf
        [ imgUrlDecoder |> D.map SaeExampleUrl
        , D.int |> D.map SaeExampleLatent
        , D.null SaeExampleMissing
        ]


parseSaeExampleResults : List SaeExampleResult -> List SaeLatent
parseSaeExampleResults results =
    let
        ( _, _, parsed ) =
            parseSaeExampleResultsHelper results [] []
    in
    parsed


parseSaeExampleResultsHelper : List SaeExampleResult -> List (Maybe String) -> List SaeLatent -> ( List SaeExampleResult, List (Maybe String), List SaeLatent )
parseSaeExampleResultsHelper unparsed urls parsed =
    case unparsed of
        [] ->
            ( [], urls, parsed )

        (SaeExampleUrl url) :: rest ->
            parseSaeExampleResultsHelper
                rest
                (urls ++ [ Just url ])
                parsed

        (SaeExampleLatent latent) :: rest ->
            parseSaeExampleResultsHelper
                rest
                (List.drop nSaeExamplesPerLatent urls)
                (parsed
                    ++ [ { latent = latent
                         , urls =
                            urls
                                |> List.take nSaeExamplesPerLatent
                                |> List.filterMap identity
                         }
                       ]
                )

        SaeExampleMissing :: rest ->
            parseSaeExampleResultsHelper
                rest
                (urls ++ [ Nothing ])
                parsed


type SegmentationResult
    = SegmentationUrl String
    | SegmentationLabels (List Int)


segmentationResultToUrl : SegmentationResult -> Maybe String
segmentationResultToUrl result =
    case result of
        SegmentationUrl url ->
            Just url

        _ ->
            Nothing


segmentationResultToLabels : SegmentationResult -> Maybe (List Int)
segmentationResultToLabels result =
    case result of
        SegmentationLabels labels ->
            Just labels

        _ ->
            Nothing


segmentationResultDecoder : D.Decoder SegmentationResult
segmentationResultDecoder =
    D.oneOf
        [ imgUrlDecoder |> D.map SegmentationUrl
        , D.list D.int |> D.map SegmentationLabels
        ]


eventIdDecoder : D.Decoder String
eventIdDecoder =
    D.field "event_id" D.string


imgUrlDecoder : D.Decoder String
imgUrlDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")


handleDuplicatedUrls : String -> String
handleDuplicatedUrls url =
    String.replace "gradio_api/gradio_api" "gradio_api" url


view : Model -> Browser.Document Msg
view model =
    { title = "Semantic Segmentation"
    , body =
        [ Html.header [] []
        , Html.main_
            []
            [ Html.div
                [ Html.Attributes.class "flex flex-row items-center" ]
                [ viewGriddedImage model model.imageUrl "Input Image"
                , viewAnimatedNeuralNetwork model
                , viewGriddedImage model model.predLabelsUrl "Predicted Semantic Segmentation"
                , viewGriddedImage model model.modifiedLabelsUrl "Modified Semantic Segmentation"
                ]
            , viewControls model
            , Html.div
                [ Html.Attributes.style "display" "flex"
                , Html.Attributes.style "flex-direction" "row"
                ]
                [ viewSaeExamples model.selectedPatchIndices model.saeLatents model.sliders
                , viewLegend (Set.fromList (model.predLabels ++ model.modifiedLabels))
                ]
            , viewErr model.err
            ]
        ]
    }


viewControls : Model -> Html.Html Msg
viewControls model =
    -- TODO: add https://thinkdobecreate.com/articles/css-animating-newly-added-element/ to new buttons
    let
        buttons =
            (if model.nPatchesExplored >= 3 then
                [ Html.button
                    [ Html.Attributes.class ""
                    , Html.Events.onClick ResetSelectedPatches
                    ]
                    [ Html.text "Clear Patches" ]
                ]

             else
                []
            )
                ++ (if
                        (model.nPatchResets
                            > 1
                            && Set.size model.selectedPatchIndices
                            > 1
                        )
                            || model.nPatchResets
                            > 3
                            || model.nImagesExplored
                            > 1
                    then
                        [ Html.button
                            [ Html.Events.onClick GetRandomExample ]
                            [ Html.text "New Image" ]
                        ]

                    else
                        []
                   )
    in
    Html.div []
        (Html.input
            [ Html.Attributes.type_ "number"
            , Html.Attributes.value (String.fromInt model.exampleIndex)
            , Html.Events.onInput SetExampleInput
            ]
            []
            :: buttons
        )


viewErr : Maybe String -> Html.Html Msg
viewErr err =
    case err of
        Just msg ->
            Html.p [ Html.Attributes.id "err-msg" ] [ Html.text msg ]

        Nothing ->
            Html.span [] []


viewGriddedImage : Model -> Maybe String -> String -> Html.Html Msg
viewGriddedImage model maybeUrl caption =
    case maybeUrl of
        Nothing ->
            Html.div [] [ Html.text "Loading..." ]

        Just url ->
            Html.div []
                [ Html.div
                    [ Html.Attributes.class "relative inline-block" ]
                    [ Html.div
                        [ Html.Attributes.class "absolute grid grid-rows-[repeat(16,_14px)] grid-cols-[repeat(16,_14px)] md:grid-rows-[repeat(16,_21px)] md:grid-cols-[repeat(16,_21px)]" ]
                        (List.map
                            (viewGridCell model.hoveredPatchIndex model.selectedPatchIndices)
                            (List.range 0 255)
                        )
                    , Html.img
                        [ Html.Attributes.class "block w-[224px] h-[224px] md:w-[336px] md:h-[336px]"
                        , Html.Attributes.src url
                        ]
                        []
                    ]
                , Html.p
                    []
                    [ Html.text caption ]
                ]


viewGridCell : Maybe Int -> Set.Set Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        classes =
            (case hovered of
                Just h ->
                    if h == self then
                        [ "border-2 border-rose-600 border-dashed" ]

                    else
                        []

                Nothing ->
                    []
            )
                ++ (if Set.member self selected then
                        [ "bg-rose-600/50" ]

                    else
                        []
                   )
    in
    Html.div
        ([ Html.Attributes.class "w-[14px] h-[14px] md:w-[21px] md:h-[21px]"
         , Html.Events.onMouseEnter (HoverPatch self)
         , Html.Events.onMouseLeave ResetHoveredPatch
         , Html.Events.onClick (ToggleSelectedPatch self)
         ]
            ++ List.map Html.Attributes.class classes
        )
        []


viewAnimatedNeuralNetwork : Model -> Html.Html Msg
viewAnimatedNeuralNetwork model =
    Html.div
        []
        [ Html.text " -> [ Neural Network ] -> " ]


viewLegend : Set.Set Int -> Html.Html Msg
viewLegend classes =
    Html.div
        [ Html.Attributes.id "legend" ]
        [ Html.p [] [ Html.text "Legend" ]
        , Html.div
            []
            (Set.toList classes
                |> List.sort
                |> List.filter (\x -> x > 0)
                |> List.map viewClassIcon
            )
        ]


viewClassIcon : Int -> Html.Html Msg
viewClassIcon class =
    let
        color =
            case Array.get (class - 1) colors of
                Just ( r, g, b ) ->
                    "rgb(" ++ String.fromInt r ++ " " ++ String.fromInt g ++ " " ++ String.fromInt b ++ ")"

                Nothing ->
                    "red"

        classname =
            case Array.get (class - 1) classnames of
                Just names ->
                    names |> List.take 2 |> String.join ", "

                Nothing ->
                    "no classname found"
    in
    Html.div
        [ Html.Attributes.class "flex flex-row gap-1 items-center" ]
        [ Html.span
            [ Html.Attributes.class "w-4 h-4"
            , Html.Attributes.style "background-color" color
            ]
            []
        , Html.span
            []
            [ Html.text (classname ++ " (class " ++ String.fromInt class ++ ")") ]
        ]


viewImageSeg : Maybe String -> String -> Html.Html Msg
viewImageSeg maybeUrl title =
    case maybeUrl of
        Just url ->
            Html.div
                []
                [ Html.img
                    [ Html.Attributes.src url
                    , Html.Attributes.style "width" "448px"
                    ]
                    []
                , Html.p [] [ Html.text title ]
                ]

        Nothing ->
            Html.div [] [ Html.text ("Loading '" ++ title ++ "'...") ]


viewSaeExamples : Set.Set Int -> List SaeLatent -> Dict.Dict Int Float -> Html.Html Msg
viewSaeExamples selected latents values =
    if List.length latents > 0 then
        Html.div []
            ([ Html.p []
                [ Html.span [ Html.Attributes.class "bg-rose-600 p-1 rounded" ] [ Html.text "These patches" ]
                , Html.text " above are like "
                , Html.span [ Html.Attributes.class "plasma-gradient text-white p-1 rounded" ] [ Html.text "these patches" ]
                , Html.text " below. (Not what you expected? Add more patches and get a larger "
                , Html.a [ Html.Attributes.href "https://simple.wikipedia.org/wiki/Sampling_(statistics)", Html.Attributes.class "text-blue-500 underline" ] [ Html.text "sample size" ]
                , Html.text ")"
                ]
             ]
                ++ List.filterMap
                    (\latent -> Maybe.map (\f -> viewSaeLatentExamples latent f) (Dict.get latent.latent values))
                    latents
            )

    else if Set.size selected > 0 then
        Html.p []
            [ Html.text "Loading similar patches..." ]

    else
        Html.p []
            [ Html.text "Click on the image above to explain model predictions." ]


viewSaeLatentExamples : SaeLatent -> Float -> Html.Html Msg
viewSaeLatentExamples latent value =
    Html.div
        [ Html.Attributes.class "flex flex-row gap-2 mt-2" ]
        (List.map viewImage latent.urls
            ++ [ Html.div
                    [ Html.Attributes.class "flex flex-col gap-2" ]
                    [ Html.input
                        [ Html.Attributes.type_ "range"
                        , Html.Attributes.min "-10"
                        , Html.Attributes.max "10"
                        , Html.Attributes.value (String.fromFloat value)
                        , Html.Events.onInput (SetSlider latent.latent)
                        ]
                        []
                    , Html.p
                        []
                        [ Html.text ("Latent 24K/" ++ String.fromInt latent.latent ++ ": " ++ String.fromFloat value) ]
                    ]
               ]
        )


viewImage : String -> Html.Html Msg
viewImage url =
    Html.img
        [ Html.Attributes.src url
        , Html.Attributes.class "max-w-36 h-auto"
        ]
        []



-- GRADIO API PARSER


eventDataParser : Parser.Parser String
eventDataParser =
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



-- CONSTANTS


nExamples =
    20210


nSaeLatents =
    3


nSaeExamplesPerLatent =
    4


randomExample : Random.Generator Int
randomExample =
    Random.int 0 (nExamples - 1)


colors : Array.Array ( Int, Int, Int )
colors =
    [ ( 51, 0, 0 ), ( 204, 0, 102 ), ( 0, 255, 0 ), ( 102, 51, 51 ), ( 153, 204, 51 ), ( 51, 51, 153 ), ( 102, 0, 51 ), ( 153, 153, 0 ), ( 51, 102, 204 ), ( 204, 255, 0 ), ( 204, 102, 0 ), ( 204, 255, 153 ), ( 102, 102, 255 ), ( 255, 204, 255 ), ( 51, 255, 0 ), ( 0, 102, 51 ), ( 102, 102, 0 ), ( 0, 0, 255 ), ( 255, 153, 204 ), ( 204, 204, 0 ), ( 0, 153, 153 ), ( 153, 102, 204 ), ( 255, 204, 0 ), ( 204, 204, 153 ), ( 255, 51, 0 ), ( 51, 51, 0 ), ( 153, 51, 51 ), ( 0, 0, 102 ), ( 102, 255, 204 ), ( 204, 51, 255 ), ( 255, 204, 204 ), ( 0, 0, 153 ), ( 0, 102, 153 ), ( 153, 0, 51 ), ( 51, 51, 102 ), ( 255, 153, 0 ), ( 204, 153, 0 ), ( 153, 102, 153 ), ( 51, 204, 204 ), ( 51, 51, 255 ), ( 153, 204, 102 ), ( 102, 204, 153 ), ( 153, 153, 204 ), ( 0, 51, 204 ), ( 204, 204, 102 ), ( 0, 51, 153 ), ( 0, 102, 0 ), ( 51, 0, 102 ), ( 153, 255, 0 ), ( 153, 255, 102 ), ( 102, 102, 51 ), ( 153, 0, 255 ), ( 204, 255, 102 ), ( 102, 0, 255 ), ( 255, 204, 153 ), ( 102, 51, 0 ), ( 102, 204, 102 ), ( 0, 102, 204 ), ( 51, 204, 0 ), ( 255, 102, 102 ), ( 153, 255, 204 ), ( 51, 204, 51 ), ( 0, 0, 0 ), ( 255, 0, 255 ), ( 153, 0, 153 ), ( 255, 204, 51 ), ( 51, 0, 51 ), ( 102, 204, 255 ), ( 153, 204, 153 ), ( 153, 102, 0 ), ( 102, 204, 204 ), ( 204, 204, 204 ), ( 255, 0, 0 ), ( 255, 255, 51 ), ( 0, 255, 102 ), ( 204, 153, 102 ), ( 204, 153, 153 ), ( 102, 51, 153 ), ( 51, 102, 0 ), ( 204, 51, 153 ), ( 153, 51, 255 ), ( 102, 0, 204 ), ( 204, 102, 153 ), ( 204, 0, 204 ), ( 102, 51, 102 ), ( 0, 153, 51 ), ( 153, 153, 51 ), ( 255, 102, 0 ), ( 255, 153, 153 ), ( 153, 0, 102 ), ( 51, 204, 255 ), ( 102, 255, 102 ), ( 255, 255, 204 ), ( 51, 51, 204 ), ( 153, 102, 51 ), ( 153, 153, 255 ), ( 51, 153, 0 ), ( 204, 0, 255 ), ( 102, 255, 0 ), ( 153, 102, 255 ), ( 204, 102, 255 ), ( 204, 0, 0 ), ( 102, 153, 255 ), ( 204, 102, 204 ), ( 204, 51, 102 ), ( 0, 255, 153 ), ( 153, 204, 204 ), ( 255, 0, 102 ), ( 102, 51, 204 ), ( 255, 51, 204 ), ( 51, 204, 153 ), ( 153, 153, 102 ), ( 153, 204, 0 ), ( 153, 102, 102 ), ( 204, 153, 255 ), ( 153, 0, 204 ), ( 102, 0, 0 ), ( 255, 51, 255 ), ( 0, 204, 153 ), ( 255, 153, 51 ), ( 0, 255, 204 ), ( 51, 102, 153 ), ( 255, 51, 51 ), ( 102, 255, 51 ), ( 0, 0, 204 ), ( 102, 255, 153 ), ( 0, 204, 255 ), ( 0, 102, 102 ), ( 102, 51, 255 ), ( 255, 0, 204 ), ( 51, 255, 153 ), ( 204, 0, 51 ), ( 153, 51, 204 ), ( 204, 102, 51 ), ( 255, 255, 0 ), ( 51, 51, 51 ), ( 0, 153, 0 ), ( 51, 255, 102 ), ( 51, 102, 255 ), ( 102, 153, 0 ), ( 102, 153, 204 ), ( 51, 0, 255 ), ( 102, 153, 153 ), ( 153, 51, 102 ), ( 204, 255, 51 ), ( 204, 204, 51 ), ( 0, 204, 51 ), ( 255, 102, 153 ), ( 204, 102, 102 ), ( 102, 0, 102 ), ( 51, 153, 204 ), ( 255, 255, 255 ), ( 0, 102, 255 ), ( 51, 102, 51 ), ( 204, 0, 153 ), ( 102, 153, 102 ), ( 102, 0, 153 ), ( 153, 255, 153 ), ( 0, 153, 102 ), ( 102, 204, 0 ), ( 0, 255, 51 ), ( 153, 204, 255 ), ( 153, 51, 153 ), ( 0, 51, 255 ), ( 51, 255, 51 ), ( 255, 102, 51 ), ( 102, 102, 204 ), ( 102, 153, 51 ), ( 0, 204, 0 ), ( 102, 204, 51 ), ( 255, 102, 255 ), ( 255, 204, 102 ), ( 102, 102, 102 ), ( 255, 102, 204 ), ( 51, 0, 153 ), ( 255, 0, 51 ), ( 102, 102, 153 ), ( 255, 153, 102 ), ( 204, 255, 204 ), ( 51, 0, 204 ), ( 0, 0, 51 ), ( 51, 255, 255 ), ( 204, 51, 0 ), ( 153, 51, 0 ), ( 51, 153, 102 ), ( 102, 255, 255 ), ( 255, 153, 255 ), ( 204, 255, 255 ), ( 204, 153, 204 ), ( 255, 0, 153 ), ( 51, 102, 102 ), ( 153, 255, 255 ), ( 255, 255, 153 ), ( 204, 51, 204 ), ( 153, 153, 153 ), ( 51, 153, 255 ), ( 51, 153, 51 ), ( 0, 153, 255 ), ( 0, 51, 51 ), ( 0, 51, 102 ), ( 153, 0, 0 ), ( 204, 51, 51 ), ( 0, 153, 204 ), ( 153, 255, 51 ), ( 255, 255, 102 ), ( 204, 204, 255 ), ( 0, 204, 102 ), ( 255, 51, 102 ), ( 0, 255, 255 ), ( 51, 153, 153 ), ( 51, 204, 102 ), ( 51, 255, 204 ), ( 255, 51, 153 ), ( 0, 51, 0 ), ( 0, 204, 204 ), ( 204, 153, 51 ) ]
        |> Array.fromList
        -- Fixed colors for example 3122
        |> Array.set 2 ( 201, 249, 255 )
        |> Array.set 4 ( 151, 204, 4 )
        |> Array.set 16 ( 54, 48, 32 )
        |> Array.set 26 ( 45, 125, 210 )
        |> Array.set 46 ( 238, 185, 2 )
        |> Array.set 72 ( 76, 46, 5 )
        |> Array.set 94 ( 12, 15, 10 )


classnames : Array.Array (List String)
classnames =
    Array.fromList
        [ [ "wall" ]
        , [ "building", "edifice" ]
        , [ "sky" ]
        , [ "floor", "flooring" ]
        , [ "tree" ]
        , [ "ceiling" ]
        , [ "road", "route" ]
        , [ "bed" ]
        , [ "windowpane", "window" ]
        , [ "grass" ]
        , [ "cabinet" ]
        , [ "sidewalk", "pavement" ]
        , [ "person", "individual" ]
        , [ "earth", "ground" ]
        , [ "door", "double door" ]
        , [ "table" ]
        , [ "mountain" ]
        , [ "plant", "flora" ]
        , [ "curtain", "drape" ]
        , [ "chair" ]
        , [ "car", "auto" ]
        , [ "water" ]
        , [ "painting", "picture" ]
        , [ "sofa", "couch" ]
        , [ "shelf" ]
        , [ "house" ]
        , [ "sea" ]
        , [ "mirror" ]
        , [ "rug", "carpet" ]
        , [ "field" ]
        , [ "armchair" ]
        , [ "seat" ]
        , [ "fence", "fencing" ]
        , [ "desk" ]
        , [ "rock", "stone" ]
        , [ "wardrobe", "closet" ]
        , [ "lamp" ]
        , [ "bathtub", "bathing tub" ]
        , [ "railing", "rail" ]
        , [ "cushion" ]
        , [ "base", "pedestal" ]
        , [ "box" ]
        , [ "column", "pillar" ]
        , [ "signboard", "sign" ]
        , [ "chest of drawers", "chest" ]
        , [ "counter" ]
        , [ "sand" ]
        , [ "sink" ]
        , [ "skyscraper" ]
        , [ "fireplace", "hearth" ]
        , [ "refrigerator", "icebox" ]
        , [ "grandstand", "covered stand" ]
        , [ "path" ]
        , [ "stairs", "steps" ]
        , [ "runway" ]
        , [ "case", "display case" ]
        , [ "pool table", "billiard table" ]
        , [ "pillow" ]
        , [ "screen door", "screen" ]
        , [ "stairway", "staircase" ]
        , [ "river" ]
        , [ "bridge", "span" ]
        , [ "bookcase" ]
        , [ "blind", "screen" ]
        , [ "coffee table", "cocktail table" ]
        , [ "toilet", "can" ]
        , [ "flower" ]
        , [ "book" ]
        , [ "hill" ]
        , [ "bench" ]
        , [ "countertop" ]
        , [ "stove", "kitchen stove" ]
        , [ "palm", "palm tree" ]
        , [ "kitchen island" ]
        , [ "computer", "computing machine" ]
        , [ "swivel chair" ]
        , [ "boat" ]
        , [ "bar" ]
        , [ "arcade machine" ]
        , [ "hovel", "hut" ]
        , [ "bus", "autobus" ]
        , [ "towel" ]
        , [ "light", "light source" ]
        , [ "truck", "motortruck" ]
        , [ "tower" ]
        , [ "chandelier", "pendant" ]
        , [ "awning", "sunshade" ]
        , [ "streetlight", "street lamp" ]
        , [ "booth", "cubicle" ]
        , [ "television receiver", "television" ]
        , [ "airplane", "aeroplane" ]
        , [ "dirt track" ]
        , [ "apparel", "wearing apparel" ]
        , [ "pole" ]
        , [ "land", "ground" ]
        , [ "bannister", "banister" ]
        , [ "escalator", "moving staircase" ]
        , [ "ottoman", "pouf" ]
        , [ "bottle" ]
        , [ "buffet", "counter" ]
        , [ "poster", "posting" ]
        , [ "stage" ]
        , [ "van" ]
        , [ "ship" ]
        , [ "fountain" ]
        , [ "conveyer belt", "conveyor belt" ]
        , [ "canopy" ]
        , [ "washer", "automatic washer" ]
        , [ "plaything", "toy" ]
        , [ "swimming pool", "swimming bath" ]
        , [ "stool" ]
        , [ "barrel", "cask" ]
        , [ "basket", "handbasket" ]
        , [ "waterfall", "falls" ]
        , [ "tent", "collapsible shelter" ]
        , [ "bag" ]
        , [ "minibike", "motorbike" ]
        , [ "cradle" ]
        , [ "oven" ]
        , [ "ball" ]
        , [ "food", "solid food" ]
        , [ "step", "stair" ]
        , [ "tank", "storage tank" ]
        , [ "trade name", "brand name" ]
        , [ "microwave", "microwave oven" ]
        , [ "pot", "flowerpot" ]
        , [ "animal", "animate being" ]
        , [ "bicycle", "bike" ]
        , [ "lake" ]
        , [ "dishwasher", "dish washer" ]
        , [ "screen", "silver screen" ]
        , [ "blanket", "cover" ]
        , [ "sculpture" ]
        , [ "hood", "exhaust hood" ]
        , [ "sconce" ]
        , [ "vase" ]
        , [ "traffic light", "traffic signal" ]
        , [ "tray" ]
        , [ "ashcan", "trash can" ]
        , [ "fan" ]
        , [ "pier", "wharf" ]
        , [ "crt screen" ]
        , [ "plate" ]
        , [ "monitor", "monitoring device" ]
        , [ "bulletin board", "notice board" ]
        , [ "shower" ]
        , [ "radiator" ]
        , [ "glass", "drinking glass" ]
        , [ "clock" ]
        , [ "flag" ]
        ]
